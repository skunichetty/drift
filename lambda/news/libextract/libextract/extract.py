import logging
import re
from typing import Any
from datetime import datetime

from bs4 import BeautifulSoup

ARTICLE_LINK_PATTERN = re.compile(r"\.\/articles\/(\w+)(?:\?\S*)?")


def valid_link(link: Any) -> bool:
    if link.text.strip() in ("", "Full Coverage"):
        return False

    url = link.get("href")
    if url is None:
        return False

    return ARTICLE_LINK_PATTERN.match(url) is not None


def _extract_time(article: Any) -> str | None:
    time = article.find("time")
    if time is not None:
        return time.get("datetime")
    return None


def _extract_authors(article: Any) -> str | None:
    authors = article.find("span", {"class": "PJK1m"})
    if authors is None:
        return None

    clean_string = authors.text[3:].replace(" &", ",")
    author_list = [author.strip().lower() for author in clean_string.split(",")]
    return "|".join([author.strip() for author in author_list])


def _extract_publisher(article: Any) -> str | None:
    publisher = article.find("div", {"class": "vr1PYe"})
    if publisher is None:
        return None
    return publisher.text.lower()


def _extract_link_info(article: Any) -> tuple[str] | tuple[None, None]:
    """Extract (ID, Title) from article if exists, else return tuple of None."""
    valid_links = list(filter(valid_link, article.find_all("a")))

    if len(valid_links) == 0:
        return None, None
    elif len(valid_links) > 1:
        raise RuntimeError("Found more than one valid link in <article> tag.")
    else:
        link = valid_links[0]
        match = ARTICLE_LINK_PATTERN.match(link.get("href"))
        return match.group(1), link.text


def extract_headlines(page_data: str) -> list[dict[str : str | datetime | None]] | None:
    headlines = []

    soup = BeautifulSoup(page_data, "html.parser")
    articles = soup.find_all("article")
    if len(articles) == 0:
        logging.warning("Found 0 <article> tags in DOM.")
        logging.warning("Review scraping algorithm, page DOM may have been updated.")
        return

    logging.debug("Found %d instances of <article> tag in webpage", len(articles))
    for index, article in enumerate(articles):
        logging.debug("Processing <article> %d...", index)

        time = _extract_time(article)
        article_id, title = _extract_link_info(article)

        if None in (time, article_id, title):
            logging.debug("-> Invalid format, skipping article")
            continue

        authors = _extract_authors(article)
        publisher = _extract_publisher(article)

        headlines.append(
            dict(
                id=article_id,
                timestamp=datetime.fromisoformat(time),
                title=title,
                publisher=publisher,
                authors=authors,
            )
        )

    if len(headlines) == 0:
        logging.warning("Found 0 <article> tags that match scraping criteria.")
        logging.warning("Review scraping algorithm, page DOM may have been updated.")

    return headlines
