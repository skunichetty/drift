import logging
import re
from random import randint, seed

import requests

URL_PATTERN = re.compile(r"(?:https?://)?(\S+)(?:\?\S*)?")

HEADER_CACHE = [
    {  # Windows Firefox
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    },
    {  # Windows Edge
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Chrome/120.0.0.0 Edg/120.0.0.0",
    },
    {  # MacOS Safari
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    },
    {  # MacOS Chrome
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    },
    {  # iPhone Safari
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.",
    },
]

SEED = 42
seed(SEED)


def get_headers() -> dict[str, str]:
    # randomly choose headers to throw off scent of trackers
    index = randint(0, len(HEADER_CACHE) - 1)
    return HEADER_CACHE[index]


def validate_url(url: str) -> bool:
    match = URL_PATTERN.match(url)
    if match is None:
        raise ValueError(f"Not a valid URL: {url}")


def fetch(url: str) -> str:
    validate_url(url)

    headers = get_headers()
    response = requests.get(url, headers=headers, stream=True)
    logging.debug("Starting page download: %s...", url[:32])

    response.raise_for_status()
    logging.debug("Successfully downloaded page.")
    logging.debug("Encoding: %s", response.encoding)
    logging.debug("Response Headers: %s", str(response.headers))
    return response.text
