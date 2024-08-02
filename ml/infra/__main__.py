import argparse
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from infra.data import load_data
from infra.data.dataset import NextClosePredictorDataset, FuturecastDataset
from infra.models.close import ClosingPricePredictor
from infra.models.futurecast import Futurecaster, AttentiveFuturecaster
from infra.train import train_model

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
        "task",
        type=str,
        choices=["close", "futurecast", "afuturecast"],
        help="Type of training task to complete",
    )
    args = parser.parse_args()

    logger.info(
        f"Training model for {args.symbol} from {args.start_date} to {args.end_date}"
    )

    try:
        train, val, pipeline = load_data(args.symbol, args.start_date, args.end_date)

        if args.task == "close":
            train_ds = NextClosePredictorDataset(train)
            val_ds = NextClosePredictorDataset(val)
        elif args.task == "futurecast":
            train_ds = FuturecastDataset(train, 48, 12)
            val_ds = FuturecastDataset(val, 48, 12)
        elif args.task == "afuturecast":
            train_ds = FuturecastDataset(train, 48, 48)
            val_ds = FuturecastDataset(val, 48, 48)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        if args.task == "close":
            model = ClosingPricePredictor(
                input_size=len(train_ds.feature_names),
                hidden_size=args.hidden_size,
                output_size=1,
            )
        elif args.task == "futurecast":
            model = Futurecaster(
                input_size=len(train_ds.feature_names),
                hidden_size=args.hidden_size,
                output_size=1,
                input_sequence_length=48,
                output_sequence_length=48,
                teacher_forcing=True,
            )
        elif args.task == "afuturecast":
            model = AttentiveFuturecaster(
                input_size=len(train_ds.feature_names),
                hidden_size=args.hidden_size,
                output_size=1,
                input_sequence_length=48,
                output_sequence_length=48,
                teacher_forcing_ratio=1,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=2
        )

        def get_custom_loss():
            loss_base = torch.nn.L1Loss()

            def custom_loss(outputs, labels):
                return 10 * loss_base(outputs, labels)

            return custom_loss

        history = train_model(
            model,
            get_custom_loss(),
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            epochs=args.epochs,
        )
    except Exception as e:
        logger.exception("Training failed")
        raise e
