import argparse
import logging
from datetime import datetime
from collections.abc import Callable

import torch
from torch.utils.data import Dataset, DataLoader

from infra.data import load_data
from infra.data.dataset import (
    NextClosePredictorDataset,
    FuturecastDataset,
    WalkerDataset,
)
from infra.models.close import ClosingPricePredictor
from infra.models.futurecast import Futurecaster, AttentiveFuturecaster
from infra.models.walker import DiscreteWalker
from infra.train import train_model
from infra.utils import raise_if_none

logger = logging.getLogger("infra")
logger.setLevel(logging.DEBUG)

stream_formatter = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] (%(process)d:%(name)s:%(lineno)d) %(message)s"
)
file_handler = logging.FileHandler(
    f"logs/train-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


def dataset_factory(
    args: argparse.Namespace,
    num_classes: int | None = None,
) -> Dataset:
    train, val, pipeline = load_data(
        args.task, args.symbol, args.start_date, args.end_date, num_classes
    )

    if args.task == "close":
        train_ds = NextClosePredictorDataset(train)
        val_ds = NextClosePredictorDataset(val)
    elif args.task in ("futurecast", "afuturecast"):
        train_ds = FuturecastDataset(
            train, args.input_sequence_length, args.output_sequence_length
        )
        val_ds = FuturecastDataset(
            val, args.input_sequence_length, args.output_sequence_length
        )
    elif args.task == "walker":
        train_ds = WalkerDataset(
            train.iloc[:10000],
            pipeline,
            raise_if_none(num_classes),
            args.input_sequence_length,
            args.output_sequence_length,
        )
        val_ds = WalkerDataset(
            val.iloc[:5000],
            pipeline,
            raise_if_none(num_classes),
            args.input_sequence_length,
            args.output_sequence_length,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return train_ds, val_ds, pipeline


def model_factory(
    args: argparse.Namespace,
    input_size: int,
    num_classes: int | None = None,
    teacher_forcing_ratio: float | None = None,
) -> torch.nn.Module:
    if args.task == "close":
        model = ClosingPricePredictor(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=1,
        )
    elif args.task == "futurecast":
        model = Futurecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=1,
            input_sequence_length=args.input_sequence_length,
            output_sequence_length=args.output_sequence_length,
            teacher_forcing=teacher_forcing_ratio is not None,
        )
    elif args.task == "afuturecast":
        model = AttentiveFuturecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=1,
            input_sequence_length=args.input_sequence_length,
            output_sequence_length=args.output_sequence_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    elif args.task == "walker":
        model = DiscreteWalker(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=raise_if_none(num_classes),
            input_sequence_length=args.input_sequence_length,
            output_sequence_length=args.output_sequence_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return model


def loss_factory(args: argparse.Namespace) -> Callable:
    if args.task == "walker":
        base = torch.nn.CrossEntropyLoss()

        def custom_loss(outputs: torch.Tensor, labels: torch.Tensor):
            # flatten across timestep dimension
            flat_outputs = outputs.flatten(end_dim=1)
            flat_labels = labels.flatten(end_dim=1).argmax(dim=1)
            return base(flat_outputs, flat_labels)

        return custom_loss
    else:

        def custom_loss(outputs, labels):
            # Penalize later predictions more
            scale = torch.arange(1, outputs.size(1) + 1, 1, device=outputs.device)
            diff = outputs - labels
            loss = torch.mean((diff * scale) ** 2)
            return loss

        return custom_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pricing model.")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Size of the hidden layer in the neural network.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("symbol", type=str, help="Stock symbol to train on.")
    parser.add_argument(
        "start_date",
        type=str,
        help="Start date for training data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "end_date",
        type=str,
        help="End date for training data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "input_sequence_length",
        type=int,
        help="Length of input sequence for the model.",
    )
    parser.add_argument(
        "output_sequence_length",
        type=int,
        help="Length of output sequence for the model.",
    )
    parser.add_argument(
        "task",
        type=str,
        choices=["close", "futurecast", "afuturecast", "walker"],
        help="Type of training task to complete",
    )
    args = parser.parse_args()

    logger.info(
        "Training %s model for %s from %s to %s",
        args.task,
        args.symbol,
        args.start_date,
        args.end_date,
    )
    logger.info(
        "Settings: Input Sequence Length: %d, Output Sequence Length: %d",
        args.input_sequence_length,
        args.output_sequence_length,
    )
    logger.info(
        "Hyperparameters: Hidden Size: %d, Learning Rate: %f", args.hidden_size, args.lr
    )

    try:
        train_dataset, val_dataset, pipeline = dataset_factory(args, num_classes=20)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        model = model_factory(
            args,
            len(train_dataset.feature_names),
            num_classes=20,
            teacher_forcing_ratio=0.5,
        )

        loss_fn = loss_factory(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=2
        )

        history = train_model(
            model,
            loss_fn,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            epochs=args.epochs,
        )
    except Exception as e:
        logger.exception("Training failed")
        raise e
