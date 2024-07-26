from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import os
import logging

import requests
import sqlalchemy as db
import sqlalchemy.dialects.postgresql as postgresql

URL = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&outputsize=full&apikey={}" 

logger = logging.getLogger("market_data_collector")

type CandleRaw = dict[str, str]

@dataclass
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

def getenv_safe(name: str):
    try:
        return os.environ[name]
    except:
        raise ValueError(f"Environment variable '{name}' is undefined")

def request_intraday(
    symbol: str,
    interval: str = "5min",
    month: str | None = None,
    extended_hours: bool = False,
) -> dict[str, CandleRaw]:
    url = URL.format(symbol, interval, getenv_safe("API_KEY"))

    if month:
        url += f"&month={month}"

    if extended_hours:
        url += f"&extended_hours={extended_hours}"

    with requests.get(url) as response:
        response.raise_for_status()
        json_data = response.json()
        return json_data["Time Series (5min)"]

def filter_data(symbol: str, intraday: dict[str, CandleRaw]) -> list[Candle]:
    candles = []
    
    today = datetime.today()
    yesterday = datetime(today.year, today.month, today.day) + timedelta(days=-1) 
    
    for time in intraday:
        time_obj = datetime.fromisoformat(time)

        if yesterday < time_obj:
            
        else:
            break

    return candles


def to_sql(data: list[Candle]):
    db_uri = getenv_safe("DB_URI")


    metadata = db.MetaData()
    table = db.Table(
        "market_data",
        metadata,
        db.Column("symbol", db.Text, primary_key=True),
        db.Column("timestamp", db.DateTime, primary_key=True),
        db.Column("open", db.FLOAT), 
        db.Column("high", db.FLOAT), 
        db.Column("low", db.FLOAT), 
        db.Column("close", db.FLOAT), 
        db.Column("volume", db.INTEGER), 
    )

    engine = db.create_engine(db_uri)
    with engine.connect() as conn:
        metadata.create_all(conn)

        query = postgresql.insert(table)
        query.on_conflict_do_nothing()

        conn.execute(query, data)
        conn.commit()

def lambda_handler(event, context):
    try:
        intraday = request_intraday(event["symbol"])
        
        today = datetime.now()
        yesterday = datetime(today.year, today.month, today.day) + timedelta(days=-1) 
        new_data = intraday.filter(pl.col("timestamp").is_between(yesterday, today))
        
        to_sql(new_data)
    except Exception as e:
       logger.error(e)
       exit(1)


if __name__ == "__main__":
    lambda_handler({"symbol": "IBM"}, None)
