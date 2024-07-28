import logging
from dataclasses import dataclass
from typing import Iterable

from libmarket.export import TableManager
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
        self.table_mgr = TableManager(config.db_uri)
        self.api = API()

    def get_new_subrequests(self, request: IntradayRequest) -> Iterable[MonthlyRequest]:
        if request.type == IntradayRequestType.DAILY:
            # always want to pull the last month of data
            yield from request.get_subrequests()
        elif request.type == IntradayRequestType.HISTORICAL:
            months_in_db = set(self.table_mgr.fetch_months(request.symbol))
            for subrequest in request.get_subrequests():
                if (
                    subrequest not in self.request_queue
                    and subrequest not in months_in_db
                ):
                    yield subrequest

    def run(self, requests: Iterable[IntradayRequest]):
        self.request_queue.load(
            self.config.request_queue_config.load.bucket_name,
            self.config.request_queue_config.load.file_name,
        )
        self.table_mgr.open()

        for request in requests:
            for subrequest in self.get_new_subrequests(request):
                self.request_queue.add(subrequest)

        if self.request_queue.empty():
            logger.info("No new requests found, shutting down orchestrator")
            return

        request = None
        try:
            request = self.request_queue.pop()
            rows = list(self.api.intraday(request.symbol, month=request.month))
            self.table_mgr.export(rows)
        except RuntimeError as e:
            if request is not None:
                self.request_queue.add(request)
            logger.debug("%s", str(e))

        self.table_mgr.close()
        self.request_queue.save(
            self.config.request_queue_config.save.bucket_name,
            self.config.request_queue_config.save.file_name,
        )
