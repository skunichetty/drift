import logging
from enum import Enum, StrEnum
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import csv

import sqlalchemy as db
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.dialects.sqlite as sqlite


def get_members(cls: Enum) -> list[str]:
    return [key for key in cls.__dict__ if key.isupper()]


class StorageMode(StrEnum):
    FILE = "file"
    SQL = "sql"
    STDOUT = "stdout"


class DatabaseBackend(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgresql"

    @classmethod
    def from_uri(cls, uri: str) -> "DatabaseBackend":
        backend = urlparse(uri).scheme
        if backend == "sqlite":
            return cls.SQLITE
        elif backend in ("postgres", "postgresql"):
            return cls.POSTGRES
        else:
            raise ValueError(f"Invalid backend: {backend}")


class FileExtension(StrEnum):
    CSV = ".csv"

    @classmethod
    def from_path(cls, path: str | PathLike) -> "FileExtension":
        path_obj = Path(path)
        extension = path_obj.suffix[1:]
        try:
            return cls[extension.upper()]
        except:
            raise ValueError(
                f"Invalid file extension found: {extension}. Must be one of {{{get_members(cls)}}}"
            )


def _write_file(rows: list[dict[str : str | datetime | None]], output: Path | None):
    if output is None:
        raise ValueError("'--output' must be specified when 'storage_mode' is 'file'")
    extension = FileExtension.from_path(output)
    if extension == FileExtension.CSV:
        if len(rows) == 0:
            logging.warning("No rows to write to file")
            return
        with output.open("w") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "timestamp", "title", "publisher", "authors"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    logging.debug("Successfully wrote '%s' to file", str(output))


def _write_stdout(rows: list[dict[str : str | datetime | None]]):
    def _to_str(value: str | datetime | None) -> str:
        if value is None:
            return ""
        elif isinstance(value, datetime):
            return value.isoformat()
        return value

    for row in rows:
        print(",".join(_to_str(row[key]) for key in row))


def _write_sql(rows: list[dict[str : str | datetime | None]], db_uri: str | None):
    if db_uri is None:
        raise ValueError("`--db_uri` must be specified when 'storage_mode' is 'sql'")

    backend = DatabaseBackend.from_uri(db_uri)
    engine = db.create_engine(db_uri)

    with engine.connect() as conn:
        meta = db.MetaData()
        news_table = db.Table(
            "news",
            meta,
            db.Column("id", db.Text, primary_key=True),
            db.Column("timestamp", db.DateTime),
            db.Column("title", db.Text),
            db.Column("publisher", db.Text),
            db.Column("authors", db.Text),
        )
        meta.create_all(conn)  # runs "CREATE TABLE IF NOT EXISTS"

        if backend == DatabaseBackend.SQLITE:
            query = sqlite.insert(news_table)
            query = query.on_conflict_do_nothing()
        elif backend == DatabaseBackend.POSTGRES:
            query = postgresql.insert(news_table)
            query = query.on_conflict_do_nothing()

        conn.execute(query, rows)
        conn.commit()


def write(
    rows: list[dict[str : str | datetime | None]] | None,
    mode: StorageMode,
    output: Path | None = None,
    db_uri: str | None = None,
):
    if mode == StorageMode.FILE:
        _write_file(rows, output)
    elif mode == StorageMode.STDOUT:
        _write_stdout(rows)
    elif mode == StorageMode.SQL:
        _write_sql(rows, db_uri)
    else:
        # this code is unreachable, is here for completeness and to future proof code
        raise NotImplementedError("Other storage modes are not implemented yet")
