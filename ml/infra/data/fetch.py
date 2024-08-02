import json
import logging
import os
import pathlib
from io import StringIO
from typing import Generator

import numpy as np
import pandas as pd
import requests

PROJECT_PATH = pathlib.Path(".")
ALPHA_VANTAGE_API = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&outputsize=full&apikey={}"
CACHE_DIR = PROJECT_PATH / ".cache"

logger = logging.getLogger(__name__)


def request_intraday(
    symbol: str,
    interval: str = "5min",
    month: str | None = None,
    extended_hours: bool = False,
):
    url = ALPHA_VANTAGE_API.format(symbol, interval, os.getenv("API_KEY"))
    if month:
        url += f"&month={month}"

    if extended_hours:
        url += "&extended_hours=true"

    logger.debug(f"Requesting intraday data for %s, month %s (%s)", symbol, month, url)

    with requests.get(url) as response:
        if response.status_code == 200:
            json_data = response.json()
            buffer = StringIO(json.dumps(json_data["Time Series (5min)"]))
            df = pd.read_json(buffer, orient="index").reset_index()
            logger.debug(f"Downloaded intraday data for %s, month %s", symbol, month)
            return df.rename(
                columns={
                    "index": "timestamp",
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume",
                }
            )
        else:
            raise Exception(f"Request failed with status code {response.status_code}")


def interp_month(
    start: np.datetime64, end: np.datetime64
) -> Generator[np.datetime64, None, None]:
    while start <= end:
        yield start
        start += np.timedelta64(1, "M")


def query_intraday(
    symbol: str,
    start_date: np.datetime64 | str,
    end_date: np.datetime64 | str,
    interval: str = "5min",
    extended_hours: bool = False,
):
    if start_date > end_date:
        raise ValueError("Start date must be before end date")

    cache_dir = CACHE_DIR / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)

    subframes = []
    for date in interp_month(start_date, end_date):
        string_date = np.datetime_as_string(date, unit="M")
        filepath = cache_dir / f"{string_date}.parquet"
        if not filepath.exists():
            subframes.append(
                request_intraday(symbol, interval, string_date, extended_hours)
            )
            subframes[-1].to_parquet(filepath)
        else:
            subframes.append(pd.read_parquet(filepath))
    return pd.concat(subframes).sort_values("timestamp").reset_index(drop=True)


def fetch_raw_features(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return query_intraday(
        symbol, np.datetime64(start_date, "M"), np.datetime64(end_date, "M")
    )
