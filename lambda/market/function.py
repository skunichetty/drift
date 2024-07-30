import logging
from typing import Any, Generator
from libmarket import orchestrator
from libmarket.fetch import getenv_safe
from libmarket.orchestrator import OrchestratorConfiguration, MarketDataOrchestrator
from libmarket.queue import (
    RequestQueueS3Configuration,
)
from libmarket.request import parse_request, IntradayRequest

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] (%(name)s) %(message)s", "%d %b %Y %H:%M:%S,%f"
)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

libmarket_logger = logging.getLogger("libmarket")
libmarket_logger.setLevel(logging.DEBUG)
libmarket_logger.addHandler(handler)
libmarket_logger.propagate = False

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
