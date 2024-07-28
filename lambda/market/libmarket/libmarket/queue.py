import json
import logging
from collections import abc, deque
from dataclasses import dataclass
from datetime import date
from io import BytesIO

from libmarket.request import MonthlyRequest

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class RequestQueue(abc.Collection[MonthlyRequest]):
    def __init__(self):
        self.requests: deque[MonthlyRequest] = deque([])
        self.index: dict[str, set[date]] = {}

    @staticmethod
    def from_str(data: str) -> "RequestQueue":
        queue = RequestQueue()
        queue.deserialize(data)
        return queue

    def add(self, request: MonthlyRequest):
        self.requests.append(request)

        self.index.setdefault(request.symbol, set())
        self.index[request.symbol].add(request.month)

    def pop(self) -> MonthlyRequest:
        request = self.requests.popleft()

        symbol_requests = self.index[request.symbol]
        symbol_requests.remove(request.month)
        return request

    def empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self.requests)

    def __contains__(self, request: MonthlyRequest) -> bool:
        if request.symbol in self.index:
            requests = self.index[request.symbol]
            return request.month in requests
        return False

    def __iter__(self) -> abc.Iterator:
        return iter(self.requests)

    def serialize(self) -> str:
        clean_requests = [request.as_dict() for request in self.requests]
        return json.dumps(clean_requests)

    def deserialize(self, data: str):
        content = json.loads(data)

        if not isinstance(content, list):
            raise ValueError(f"Expected list, received {type(content)}")

        for index, request_data in enumerate(content):
            if not isinstance(request_data, dict):
                raise ValueError(
                    f"Expected dict at index {index} in list, received {type(content)}"
                )
            self.add(MonthlyRequest.from_dict(request_data))


class S3BackedRequestQueue(RequestQueue):
    def __init__(self):
        super().__init__()
        self.s3_client = boto3.client("s3")

    def load(self, bucket_name: str, file_name: str):
        logger.debug("Loading request queue from S3 - %s:%s", bucket_name, file_name)
        buffer = BytesIO()
        try:
            self.s3_client.download_fileobj(bucket_name, file_name, buffer)
            self.deserialize(buffer.read().decode())
        except (ClientError, UnicodeError) as e:
            logger.debug("Unable to load request queue from s3: %s", str(e))

    def save(self, bucket_name: str, file_name: str):
        logger.debug("Saving request queue to S3 - %s:%s", bucket_name, file_name)
        try:
            buffer = BytesIO(self.serialize().encode())
            self.s3_client.upload_fileobj(buffer, bucket_name, file_name)
        except (ClientError, UnicodeError) as e:
            logger.error("Unable to save request queue to S3: %s", str(e))


@dataclass
class RequestQueueS3BucketConfiguration:
    bucket_name: str
    file_name: str


@dataclass
class RequestQueueS3Configuration:
    def __init__(
        self, input_bucket: str, input_file: str, output_bucket: str, output_file: str
    ):
        self.load = RequestQueueS3BucketConfiguration(input_bucket, input_file)
        self.save = RequestQueueS3BucketConfiguration(output_bucket, output_file)
