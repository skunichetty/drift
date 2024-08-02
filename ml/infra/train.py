from pathlib import Path
from typing import Callable, Union
import logging

import numpy as np
import seaborn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


seaborn.set_theme()


def save_model(
    epoch: int,
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
):
    """
    Save the model to a file.

    Args:
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to be saved.
        checkpoint_dir (Path): The directory to save the model to.
    """
    checkpoint_dir = Path(checkpoint_dir) / model.__class__.__name__
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"model_{epoch}.pt"
    torch.save(model.state_dict(), checkpoint_file)

    logger.info("Saving model (epoch %d) to %s", epoch, str(checkpoint_file))


def load_model(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    epoch: int,
    map_location: str = "cpu",
):
    """
    Load the model from a file.

    Args:
        model (torch.nn.Module): The model to be loaded.
        checkpoint_dir (Union[str, Path]): The directory to load the model from.
        epoch (int): The epoch number of the model to load.
        map_location (str, optional): The device to load the model on. Defaults to "cpu".

    Raises:
        FileNotFoundError: If the specified model file does not exist.

    Returns:
        None
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_file = checkpoint_dir / model.__class__.__name__ / f"model_{epoch}.pt"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model at epoch {epoch} in directory {str(model_file.parent)} does not exist."
        )

    model.load_state_dict(
        torch.load(
            model_file, map_location=torch.device(map_location), weights_only=True
        )
    )


def evaluate(
    model: torch.nn.Module, criterion: Callable, loader: DataLoader
) -> tuple[float]:
    """
    Evaluate the performance of a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (Callable): The loss function used for evaluation.
        loader (DataLoader): The data loader containing the evaluation data.

    Returns:
        tuple[float]: A tuple containing the average loss.

    """
    model.eval()
    total = 0
    with torch.no_grad():
        loss = 0.0
        for X, y in loader:
            outputs = model(X.to(device)).to("cpu")
            loss += criterion(outputs, y).detach().sum().item()
            total += len(y)
    model.train()
    return loss / total


def train_model(
    model: torch.nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    save_checkpoint: bool = True,
    checkpoint_dir: Union[Path, str] = "checkpoints",
) -> dict[str, list[float]]:
    """
    Train a given model using the specified criterion, optimizer, and data loaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (Callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler used for adjusting the learning rate.
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        epochs (int, optional): The number of training epochs. Defaults to 10.
        save_checkpoint (bool, optional): Whether to save model checkpoints. Defaults to True.
        checkpoint_dir (Union[Path, str], optional): The directory to save the model checkpoints. Defaults to "checkpoints".

    Returns:
        dict[str, list[float]]: A dictionary containing the training and validation losses and accuracies.
    """
    logger.info(f"Starting training, using {device} device")

    train_losses, val_losses = [], []
    model = model.to(device)

    with logging_redirect_tqdm():
        for epoch in range(epochs):
            model.train()
            epoch_train_losses = []

            pbar = tqdm(train_loader)
            for X, y in pbar:
                optimizer.zero_grad()
                outputs = model(X.to(device), y=y.to(device))
                loss = criterion(outputs, y.to(device))
                loss.backward()
                optimizer.step()

                cpu_loss = loss.to("cpu").item()
                epoch_train_losses.append(cpu_loss)

                pbar.update(1)
                pbar.set_description(
                    f"Batch Loss: {cpu_loss:.2f} , Average Loss: {np.mean(epoch_train_losses):.2f}"
                )

            train_losses.append(np.mean(epoch_train_losses))
            epoch_train_losses.clear()
            val_loss = evaluate(model, criterion, val_loader)
            val_losses.append(val_loss)

            if save_checkpoint:
                save_model(epoch, model, Path(checkpoint_dir))

            scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}: Loss - (Train {train_losses[-1]:.2f}/Val {val_losses[-1]:.2f})"
            )

    return {
        "loss": {
            "train": train_losses,
            "val": val_losses,
        },
    }
