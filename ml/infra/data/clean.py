import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


def is_regular_hours(data: pd.DataFrame) -> pd.DataFrame:
    time_range = pd.DatetimeIndex(data["timestamp"])
    regular_hours = time_range.indexer_between_time("09:30", "16:00")
    frame = pd.DataFrame(
        {"regular_hours": data["timestamp"].index.isin(regular_hours) * 1}
    )
    return frame


def _ts_encode(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    radians = 2 * np.pi * series / period
    return pd.DataFrame(
        {f"sin_{name}": np.sin(radians), f"cos_{name}": np.cos(radians)}
    )


def ts_encode(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = df["timestamp"]

    time_subunits = [timestamp.dt.minute, timestamp.dt.hour, timestamp.dt.month]
    periods = [60, 24, 12]
    names = ["minute", "hour", "month"]

    zipped_iter = zip(time_subunits, periods, names)
    subframes = [
        _ts_encode(subunit, period, name) for subunit, period, name in zipped_iter
    ]

    frame = pd.concat(subframes, axis=1)
    frame["year"] = timestamp.dt.year
    frame["day"] = timestamp.dt.day
    frame["timestamp"] = timestamp
    return frame


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns
    frame = df.assign(
        **{f"delta_{col}": df[col].diff().fillna(df[col]) for col in columns}
    )
    return frame


def get_pipeline():
    pipeline = Pipeline(
        [
            (
                "ts_encoder",
                ColumnTransformer(
                    [
                        ("ts_encode", FunctionTransformer(ts_encode), ["timestamp"]),
                        (
                            "regular_hours",
                            FunctionTransformer(is_regular_hours),
                            ["timestamp"],
                        ),
                        (
                            "delta",
                            FunctionTransformer(add_delta),
                            ["open", "close", "high", "low"],
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
            (
                "scale",
                ColumnTransformer(
                    [
                        (
                            "scale",
                            StandardScaler(),
                            ["volume"],
                        )
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
        ]
    )
    return pipeline
