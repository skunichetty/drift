from datetime import date, datetime
from typing import Any, Generator

import sqlalchemy as db
import sqlalchemy.dialects.postgresql as postgresql


class TableManager:
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
        self.table_created = False

    def open(self):
        self._conn = self.engine.connect()

    def __enter__(self):
        self.open()

    def __exit__(self, exc, value, tb):
        self.close()

    def export(self, rows: list[dict[str, Any]]):
        if not self.table_created:
            self.metadata.create_all(self.conn)
            self.table_created = True

        query = postgresql.insert(self.table)
        query = query.on_conflict_do_nothing()
        self.conn.execute(query, rows)

    def fetch_months(self, symbol: str) -> Generator[date, None, None]:
        # TODO: consider caching these results
        query = (
            db.select(
                db.extract("YEAR", self.table.c.timestamp).label("year"),
                db.extract("MONTH", self.table.c.timestamp).label("month"),
            )
            .where(self.table.c.symbol == symbol)
            .distinct()
        )

        for raw_date in self.conn.execute(query):
            yield datetime.strptime(f"{raw_date[0]}-{raw_date[1]}", "%Y-%m").date()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()

    @property
    def conn(self) -> db.Connection:
        if self._conn is None:
            raise RuntimeError("Connection is closed - call open() to start connection")
        return self._conn
