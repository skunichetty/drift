import logging
import logging
from argparse import ArgumentParser
from pathlib import Path

from libextract.export import FileExtension, StorageMode, get_members, write
from libextract.extract import extract_headlines
from libextract.fetch import fetch


def build_parser() -> ArgumentParser:
    parser = ArgumentParser("news", description="Extract current headlines")
    parser.add_argument(
        "url",
        help="URL to download.",
        type=str,
    )

    parser.add_argument(
        "storage_mode",
        help="Specifies how to store headline data. Either 'file','sql', or 'stdout' (case insensitive). "
        "If 'file' specified, then the '--output' option should be provided. "
        "If 'sql' is specified, the '--db_uri' option should be enabled. "
        "Defaults to 'stdout'",
        choices=[StorageMode[member] for member in get_members(StorageMode)],
        type=StorageMode,
        default=StorageMode.STDOUT,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file to save the headline data to. "
        f"Ignored if 'storage_mode' is not 'file'. Valid file formats are {{{', '.join(get_members(FileExtension))}}}",
        type=Path,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose logging for script. Defaults to False",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--db_uri",
        help="URI of database to write headlines to. Stores data under the 'news' table. "
        "Ignored if 'storage_mode' is not 'sql'."
        "See http://tools.ietf.org/html/rfc3986 for URI details.",
        type=str,
    )
    return parser


def run_extract(
    url: str,
    mode: StorageMode,
    output: Path | None,
    db_uri: str | None,
    verbose: bool = False,
):
    if verbose:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

    try:
        page_data = fetch(url)
        headlines = extract_headlines(page_data)
        if headlines is None:
            raise RuntimeError("No headlines extracted from page. Exiting.")
        write(headlines, mode, output, db_uri)
    except Exception as e:
        logging.error(e)
        exit(1)
