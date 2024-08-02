import torch
import torch.nn.functional as F

from infra.models.rnn import LSTMCell


class ClosingPricePredictor(torch.nn.Module):
    """
    A class representing a closing price predictor model.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state in the LSTM cell.
        output_size (int): The size of the output.
        teacher_force (bool, optional): Whether to use teacher forcing during training. Defaults to True.
    """

    def __init__(self, input_size, hidden_size, output_size, teacher_force=True):
        super(ClosingPricePredictor, self).__init__()
        self.lstm = LSTMCell(input_size, hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.teacher_force = teacher_force

    def init_weights(self):
        """
        Initialize the weights of the MLP layers using Xavier normal initialization
        and set the biases to zero.
        """
        for weight in [self.mlp[-1], self.mlp[3]]:
            torch.nn.init.xavier_normal_(weight.weight)
            torch.nn.init.zeros_(weight.bias)

    def forward(self, x, c=None, h=None):
        """
        Forward pass of the closing price predictor model.

        Args:
            x (torch.Tensor): The input tensor of shape (N, L, I), where N is the batch size,
                L is the sequence length, and I is the input size.
            c (torch.Tensor, optional): The cell state tensor of shape (N, hidden_size).
                Defaults to None.
            h (torch.Tensor, optional): The hidden state tensor of shape (N, hidden_size).
                Defaults to None.

        Returns:
            torch.Tensor: The predicted closing prices tensor of shape (N, output_size).
        """
        N, L, I = x.shape

        if c is None:
            c = torch.zeros(N, self.hidden_size).to(x.device)

        if h is None:
            h = torch.zeros(N, self.hidden_size).to(x.device)

        for time in range(L):
            h, c = self.lstm(x[:, time, :], c, h)
        return self.mlp(h).squeeze()
