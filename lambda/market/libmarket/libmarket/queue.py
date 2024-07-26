from datetime import date, datetime
from dataclasses import dataclass
from collections import deque, abc
import json

MONTH_FMT_STRING = "%Y-%m"

@dataclass
class Request:
    symbol: str
    month: date  

    @staticmethod
    def from_string(symbol: str, month: str) -> "Request":
        date_obj = datetime.strptime(month, MONTH_FMT_STRING) 
        return Request(symbol, date_obj.date())

    @staticmethod
    def from_dict(rep: dict[str, str]) -> "Request":
        try:
            return Request.from_string(rep["symbol"], rep["month"]) 
        except KeyError:
            raise ValueError(f"Invalid dictionary representation: {rep}")
        
    def as_dict(self) -> dict[str, str]:
        month_str = self.month.strftime(MONTH_FMT_STRING)
        return {"symbol": self.symbol, "month": month_str}

class RequestQueue(abc.Collection[Request]):
    def __init__(self):
        self.requests : deque[Request] = deque([])
        self.index : dict[str, set[date]] = {}

    def add(self, request: Request): 
        self.requests.append(request)
        
        self.index.setdefault(request.symbol, set())
        self.index[request.symbol].add(request.month)

    def pop(self) -> Request:
        request = self.requests.popleft()
        
        symbol_requests = self.index[request.symbol]
        symbol_requests.remove(request.month)
        return request

    def empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self.requests)

    def __contains__(self, request: Request) -> bool:
        if request.symbol in self.index:
            requests = self.index[request.symbol]
            return request.month in requests
        return False

    def __iter__(self) -> abc.Iterator:
        return iter(self.requests)

    def serialize(self) -> str:
        clean_requests = [request.as_dict() for request in self.requests]
        return json.dumps(clean_requests)

    @staticmethod
    def deserialize(data: str) -> "RequestQueue":
        queue = RequestQueue()
        content = json.loads(data)

        if not isinstance(content, list):
            raise ValueError(f"Expected list, received {type(content)}")

        for index, request_data in enumerate(content):
            if not isinstance(request_data, dict):
                raise ValueError(f"Expected dict at index {index} in list, received {type(content)}")
            queue.add(Request.from_dict(request_data))

        return queue
