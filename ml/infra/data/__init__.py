from infra.data.clean import get_pipeline
from infra.data.fetch import fetch_raw_features
from infra.data.dataset import train_val_split

import logging

logger = logging.getLogger(__name__)


def load_data(
    task: str,
    symbol: str,
    start_date: str,
    end_date: str,
    num_classes: int | None = None,
):
    logger.info("Loading dataset for task %s", task)
    raw_features = fetch_raw_features(symbol, start_date, end_date)
    train, val = train_val_split(raw_features, train_prop=0.8)

    pipeline = get_pipeline(task, num_classes)
    logger.info("Fitting and transforming training data")
    train = pipeline.fit_transform(train)
    logger.info("Transforming validation data")
    val = pipeline.transform(val)

    return train, val, pipeline
