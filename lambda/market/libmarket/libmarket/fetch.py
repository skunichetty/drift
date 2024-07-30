import os
from datetime import date, datetime
from typing import Any, Generator

import requests
import logging

API_REQUEST_LIMIT = 25

logger = logging.getLogger(__name__)


def getenv_safe(name: str):
    try:
        return os.environ[name]
    except:
        raise ValueError(f"Environment variable '{name}' is undefined")


class API:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.request_count = 0
        self.session = None

    def build_url(self, endpoint: str, **kwargs: str) -> str:
        base = f"{self.BASE_URL}?function={endpoint}&"
        arguments = [f"{key}={kwargs[key]}" for key in kwargs]
        return base + "&".join(arguments)

    def request(self, url: str) -> Any:
        if self.request_count < API_REQUEST_LIMIT:
            logger.debug("Querying API: %s", url)

            if self.session is None:
                self.session = requests.session()

            with self.session.get(url) as response:
                response.raise_for_status()
                return response.json()

        else:
            raise RuntimeError("API Call Limit Reached - try again in 24 hours.")

    def intraday(
        self,
        symbol: str,
        interval: str = "5min",
        month: date | None = None,
        extended_hours: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        args = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
        }

        if month:
            args["month"] = month.strftime("%Y-%m")

        if extended_hours:
            args["extended_hours"] = "true"

        args["apikey"] = getenv_safe("API_KEY")

        url = self.build_url("TIME_SERIES_INTRADAY", **args)
        intraday_json = self.request(url)

        intraday_series = intraday_json[f"Time Series ({interval})"]
        for timestamp in intraday_series:
            yield {
                "symbol": symbol,
                "timestamp": datetime.fromisoformat(timestamp),
                "open": float(intraday_series[timestamp]["1. open"]),
                "high": float(intraday_series[timestamp]["2. high"]),
                "low": float(intraday_series[timestamp]["3. low"]),
                "close": float(intraday_series[timestamp]["4. close"]),
                "volume": int(intraday_series[timestamp]["5. volume"]),
            }
