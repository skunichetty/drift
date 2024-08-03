from typing import TypeVar

T = TypeVar("T")


def raise_if_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Value is None")
    return value
