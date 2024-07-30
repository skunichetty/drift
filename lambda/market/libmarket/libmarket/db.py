import logging
from datetime import date, datetime
from typing import Any, Generator

import sqlalchemy as db
import sqlalchemy.dialects.postgresql as postgresql

logger = logging.getLogger(__name__)


class MarketDataTableAPI:
    def __init__(self, uri: str):
        self.engine = db.create_engine(uri)
        self.metadata = db.MetaData()

        self._conn = None

        self.table = db.Table(
            "market_data",
            self.metadata,
            db.Column("symbol", db.Text, primary_key=True),
            db.Column("timestamp", db.DateTime, primary_key=True),
            db.Column("open", db.FLOAT),
            db.Column("high", db.FLOAT),
            db.Column("low", db.FLOAT),
            db.Column("close", db.FLOAT),
            db.Column("volume", db.INTEGER),
        )

    def open(self):
        logger.debug("Starting database connection: %s", self.engine.url)
        self._conn = self.engine.connect()
        logger.debug("Successfully connected to database")
        logger.debug("Creating 'market_data' table if not already existing")
        self.metadata.create_all(self.conn)
        logger.debug("Successfully created 'market_data' table")

    def export(self, rows: list[dict[str, Any]]):
        logger.debug("Writing %d rows to 'market_data' table", len(rows))
        query = postgresql.insert(self.table)
        query = query.on_conflict_do_nothing()
        self.conn.execute(query, rows)
        logger.debug("Successfully wrote %d rows to 'market_data' table", len(rows))

    def fetch_months(self, symbol: str) -> Generator[date, None, None]:
        logger.debug("Querying stored months for '%s' from 'market_data' table", symbol)
        query = (
            db.select(
                db.extract("YEAR", self.table.c.timestamp).label("year"),
                db.extract("MONTH", self.table.c.timestamp).label("month"),
            )
            .where(self.table.c.symbol == symbol)
            .distinct()
        )

        cur = self.conn.execute(query)
        logger.debug("Successfully fetched stored months for '%s'", symbol)
        for raw_date in cur.fetchall():
            yield datetime.strptime(f"{raw_date[0]}-{raw_date[1]}", "%Y-%m").date()

    def commit(self):
        self.conn.commit()
        logger.debug("Successfully committed session results to DB")

    def rollback(self):
        self.conn.rollback()
        logger.debug("Successfully rolled back session results")

    def close(self):
        if self._conn is not None:
            logger.debug("Closing connection to DB")
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()

    @property
    def conn(self) -> db.Connection:
        if self._conn is None:
            raise RuntimeError("Connection is closed - call open() to start connection")
        return self._conn

    def __enter__(self):
        self.open()

    def __exit__(self, exc, value, tb):
        if exc is not None:
            logger.debug("Exception raised by table API - initiating rollback")
            self.rollback()
