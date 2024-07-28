from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Generator, Iterable, override

from dateutil.relativedelta import relativedelta

MONTH_FMT_STRING = "%Y-%m"


def date_from_string(month: str) -> date:
    date_obj = datetime.strptime(month, MONTH_FMT_STRING)
    return date_obj.date()


@dataclass
class MonthlyRequest:
    symbol: str
    month: date

    @staticmethod
    def from_string(symbol: str, month: str) -> "MonthlyRequest":
        return MonthlyRequest(symbol, date_from_string(month))

    @staticmethod
    def from_dict(rep: dict[str, str]) -> "MonthlyRequest":
        try:
            return MonthlyRequest.from_string(rep["symbol"], rep["month"])
        except KeyError:
            raise ValueError(f"Invalid dictionary representation: {rep}")

    def as_dict(self) -> dict[str, str]:
        month_str = self.month.strftime(MONTH_FMT_STRING)
        return {"symbol": self.symbol, "month": month_str}


class IntradayRequestType(StrEnum):
    DAILY = "daily"
    HISTORICAL = "historical"


class IntradayRequest(ABC):
    def __init__(self, symbol: str):
        self.symbol = symbol

    @abstractmethod
    def get_subrequests(self) -> Iterable[MonthlyRequest]:
        pass

    @property
    @abstractmethod
    def type(self) -> IntradayRequestType:
        pass


class DailyIntradayRequest(IntradayRequest):
    @override
    def get_subrequests(self) -> list[MonthlyRequest]:
        # get only the most recent month of data
        return [MonthlyRequest(self.symbol, datetime.today().date())]

    @property
    @override
    def type(self) -> IntradayRequestType:
        return IntradayRequestType.DAILY


class HistoricalIntradayRequest(IntradayRequest):
    def __init__(self, symbol: str, start_date: date, end_date: date):
        super().__init__(symbol)
        self.start_date = start_date
        self.end_date = end_date

    @override
    def get_subrequests(self) -> Generator[MonthlyRequest, None, None]:
        delta = relativedelta(month=1)

        current = self.start_date
        while current <= self.end_date:
            yield MonthlyRequest(self.symbol, current)
            current += delta

    @property
    @override
    def type(self) -> IntradayRequestType:
        return IntradayRequestType.HISTORICAL


def parse_request(data: dict[str, Any]) -> IntradayRequest:
    try:
        request_type = data["type"]
        symbol = data["symbol"]
    except KeyError as e:
        raise ValueError(f"Missing required key '{str(e)}'")

    if request_type == "daily":
        return DailyIntradayRequest(symbol)
    elif request_type == "historical":
        try:
            options = data["options"]
        except KeyError as e:
            raise ValueError(
                "Missing key options which is required when request type is 'historical'"
            )

        try:
            start_date = date_from_string(options["start_date"])
            end_date = date_from_string(options["end_date"])
        except KeyError as e:
            raise ValueError(
                f"Missing option '{str(e)}', which is required when request type is 'historical'"
            )

        return HistoricalIntradayRequest(symbol, start_date, end_date)
    else:
        raise ValueError(f"Unknown request type '{request_type}'")
