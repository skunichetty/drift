import logging
from typing import Any, Generator
from libmarket import orchestrator
from libmarket.fetch import getenv_safe
from libmarket.orchestrator import OrchestratorConfiguration, MarketDataOrchestrator
from libmarket.queue import (
    RequestQueueS3Configuration,
)
from libmarket.request import parse_request, IntradayRequest


logger = logging.getLogger(__name__)

BUCKET_NAME = "drift"
RQ_FILENAME = "mdata-rq.json"


"""
Event schema:
{
    request: [
        {
            // Download historical data (month by month basis) 
            symbol: "AAA",
            type: "historical",
            options: {
                start_date: "2020-01",
                end_date: "2024-07",
            }
        },
        {
            // Download most recent day of data
            symbol: "AAA",
            type: "day"
        }
    ]
}
"""

request_queue_config = RequestQueueS3Configuration(
    BUCKET_NAME, RQ_FILENAME, BUCKET_NAME, RQ_FILENAME
)
config = OrchestratorConfiguration(request_queue_config, getenv_safe("DB_URI"))
orchestrator = MarketDataOrchestrator(config)


def read_requests(event: dict[str, Any]) -> Generator[IntradayRequest, None, None]:
    raw_requests = event["requests"]
    if not isinstance(raw_requests, list):
        raise ValueError("Invalid event payload - 'requests' key must be a list")

    for raw_request in raw_requests:
        yield parse_request(raw_request)


def lambda_handler(event, context):
    try:
        orchestrator.run(read_requests(event))
    except Exception as e:
        logger.error(e)
        exit(1)


if __name__ == "__main__":
    lambda_handler({"symbol": "IBM"}, None)
