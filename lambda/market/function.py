import logging
from typing import Any, Generator
from libmarket import orchestrator
from libmarket.fetch import getenv_safe
from libmarket.orchestrator import OrchestratorConfiguration, MarketDataOrchestrator
from libmarket.queue import (
    RequestQueueS3Configuration,
)
from libmarket.request import parse_request, IntradayRequest

logging.getLogger().propagate = False  # prevent root logger settings from propagating

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(name)s - %(levelname)s] - %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

libmarket_logger = logging.getLogger("libmarket")
libmarket_logger.setLevel(logging.DEBUG)
libmarket_logger.addHandler(handler)

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

BUCKET_NAME = getenv_safe("BUCKET_NAME")
REQUEST_QUEUE_FNAME = getenv_safe("REQUEST_QUEUE_FNAME")

config = OrchestratorConfiguration(
    RequestQueueS3Configuration(
        BUCKET_NAME, REQUEST_QUEUE_FNAME, BUCKET_NAME, REQUEST_QUEUE_FNAME
    ),
    getenv_safe("DB_URI"),
)

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
        logger.error(e.__class__.__name__)
        logger.error(e, exc_info=True)
        exit(1)
