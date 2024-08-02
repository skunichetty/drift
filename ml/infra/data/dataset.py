import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NextClosePredictorDataset(Dataset):
    """
    A custom PyTorch dataset for predicting the next closing price.

    Args:
        data (pd.DataFrame): The input data containing the financial features.
        sequence_length (int): The length of the input sequence. Defaults to 128.

    Attributes:
        sequence_length (int): The length of the input sequence.
        data (pd.DataFrame): The input data containing the financial features.
        feature_names (list): The list of feature names.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the features and label for a given index.

    """

    def __init__(self, data: pd.DataFrame, sequence_length=128):
        self.sequence_length = sequence_length
        self.data = data
        self.feature_names = list(data.columns.difference(["timestamp", "year", "day"]))

    def __len__(self):
        return self.data.shape[0] - self.sequence_length

    def __getitem__(self, idx):
        if idx + self.sequence_length >= self.data.shape[0]:
            raise IndexError("Index out of range")

        features = (
            self.data[self.feature_names]
            .iloc[idx : idx + self.sequence_length]
            .to_numpy()
            .astype(np.float32)
        )
        labels = self.data["close"].iloc[idx + self.sequence_length].astype(np.float32)
        return features, labels


class FuturecastDataset(Dataset):
    """
    A custom PyTorch dataset for futurecasting the closing price.

    Args:
        data (pd.DataFrame): The input data containing the financial features.
        sequence_length (int): The length of the input sequence. Defaults to 128.

    Attributes:
        sequence_length (int): The length of the input sequence.
        data (pd.DataFrame): The input data containing the financial features.
        feature_names (list): The list of feature names.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the features and label for a given index.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        in_sequence_length=72,
        out_sequence_length=12,
    ):
        self.in_sequence_length = in_sequence_length
        self.out_sequence_length = out_sequence_length
        self.data = data
        self.feature_names = list(
            data.columns.difference(
                ["timestamp", "year", "day", "open", "close", "high", "low"]
            )
        )

    def __len__(self):
        return (
            self.data.shape[0]
            - (self.out_sequence_length + self.in_sequence_length)
            - 1
        )

    def __getitem__(self, idx):
        adj_idx = idx + 1  # never return the first row
        input_end_index = adj_idx + self.in_sequence_length
        final_index = adj_idx + self.in_sequence_length + self.out_sequence_length

        if final_index >= self.data.shape[0]:
            raise IndexError("Index out of range")

        features = (
            self.data[self.feature_names]
            .iloc[adj_idx:input_end_index]
            .to_numpy()
            .astype(np.float32)
        )
        labels = (
            self.data["delta_close"]
            .iloc[input_end_index:final_index]
            .to_numpy()
            .astype(np.float32)
        )
        return features, labels

    def adjust_delta(self, feature_name: str, index: int) -> pd.Series:
        delta_name = f"delta_{feature_name}"
        columns = self.data.columns

        if feature_name not in columns:
            raise ValueError("Feature name not found in dataset")
        if delta_name not in columns:
            raise ValueError("Delta feature not found in dataset")

        return self.data[feature_name].iloc[index]


def train_val_split(
    data: pd.DataFrame, train_prop: float = 0.8, drop_index: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the given DataFrame into training and testing sets.

    Parameters:
        data (pd.DataFrame): The input DataFrame to be split.
        train_prop (float): The proportion of data to be used for training. Default is 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing sets.
    """
    train_size = int(data.shape[0] * train_prop)
    train = data.iloc[:train_size].reset_index(drop=drop_index)
    val = data.iloc[train_size:].reset_index(drop=drop_index)
    return train, val
