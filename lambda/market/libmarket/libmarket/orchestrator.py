import logging
from dataclasses import dataclass
from typing import Iterable

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

    def get_new_subrequests(self, request: IntradayRequest) -> Iterable[MonthlyRequest]:
        logger.debug(
            "Getting subrequests for symbol %s (%s)", request.symbol, request.type
        )
        if request.type == IntradayRequestType.DAILY:
            logger.debug("Generating subrequest for most recent month")
            yield from request.get_subrequests()
        elif request.type == IntradayRequestType.HISTORICAL:
            months_in_db = set(self.table_api.fetch_months(request.symbol))
            logger.debug("Found %d months already in DB", len(months_in_db))
            for subrequest in request.get_subrequests():
                if (
                    subrequest not in self.request_queue
                    and subrequest.as_dict()["month"] not in months_in_db
                ):
                    yield subrequest

    def run(self, requests: Iterable[IntradayRequest]):
        logger.debug("Starting orchestration job")

        self.request_queue.load(
            self.config.request_queue_config.load.bucket_name,
            self.config.request_queue_config.load.file_name,
        )
        initial_queue_size = len(self.request_queue)

        with self.table_api:
            for request in requests:
                for subrequest in self.get_new_subrequests(request):
                    self.request_queue.add(subrequest)

            final_queue_size = len(self.request_queue)

            if self.request_queue.empty():
                logger.debug("No new requests found, shutting down orchestrator")
                return

            logger.debug(
                "Added %d new requests to queue. Approximate processing time: %f days",
                final_queue_size - initial_queue_size,
                min(1, final_queue_size / 25),
            )

            request = None
            try:
                request = self.request_queue.pop()
                rows = list(
                    self.mdata_api.intraday(symbol=request.symbol, month=request.month)
                )
                self.table_api.export(rows)
            except RuntimeError as e:
                if request is not None:
                    self.request_queue.add(request)
                logger.debug(e)

            self.table_api.commit()

        self.request_queue.save(
            self.config.request_queue_config.save.bucket_name,
            self.config.request_queue_config.save.file_name,
        )
