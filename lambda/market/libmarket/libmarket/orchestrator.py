import logging
from dataclasses import dataclass
from itertools import chain
from typing import Generator, Iterable

from libmarket.db import MarketDataTableAPI
from libmarket.fetch import API
from libmarket.queue import RequestQueueS3Configuration, S3BackedRequestQueue
from libmarket.request import IntradayRequest, IntradayRequestType, MonthlyRequest

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfiguration:
    request_queue_config: RequestQueueS3Configuration
    db_uri: str


class MarketDataOrchestrator:
    def __init__(self, config: OrchestratorConfiguration):
        self.request_queue = S3BackedRequestQueue()
        self.config = config
        self.table_api = MarketDataTableAPI(config.db_uri)
        self.mdata_api = API()

        self.request_queue.load(
            self.config.request_queue_config.load.bucket_name,
            self.config.request_queue_config.load.file_name,
        )

    def _filter_from_db(
        self, request: IntradayRequest
    ) -> Generator[MonthlyRequest, None, None]:
        months_in_db = set(self.table_api.fetch_months(request.symbol))
        logger.debug(
            "Found %d months in DB for symbol %s", len(months_in_db), request.symbol
        )
        for subrequest in request.get_subrequests():
            if subrequest.as_dict()["month"] not in months_in_db:
                yield subrequest

    def filter_subrequests(self, request: IntradayRequest) -> Iterable[MonthlyRequest]:
        logger.debug("Getting subrequests for %s (%s)", request.symbol, request.type)

        stream = request.get_subrequests()
        if request.type == IntradayRequestType.HISTORICAL:
            # for historical data, avoid querying data stored to DB
            # for daily request, will ALWAYS redownload most recent day
            stream = self._filter_from_db(request)

        for subrequest in stream:
            if subrequest not in self.request_queue:
                yield subrequest

    def download_and_export(self):
        request = None
        try:
            while not self.request_queue.empty():
                request = self.request_queue.pop()
                print(request)
                rows = list(
                    self.mdata_api.intraday(symbol=request.symbol, month=request.month)
                )
                self.table_api.export(rows)
        except RuntimeError as e:
            logger.debug(e)
            if request is not None:
                self.request_queue.add(request)

    def run(self, requests: Iterable[IntradayRequest]):
        logger.debug("Starting orchestration job")

        initial_queue_size = len(self.request_queue)

        with self.table_api:
            self.request_queue.extend(chain(*map(self.filter_subrequests, requests)))
            final_queue_size = len(self.request_queue)

            if self.request_queue.empty():
                logger.debug("No new requests found, shutting down orchestrator")
                return

            logger.debug(
                "Added %d new requests to queue. Approximate processing time: %f days",
                final_queue_size - initial_queue_size,
                max(1, final_queue_size / 25),
            )

            self.download_and_export()
            self.table_api.commit()

        self.request_queue.save(
            self.config.request_queue_config.save.bucket_name,
            self.config.request_queue_config.save.file_name,
        )
