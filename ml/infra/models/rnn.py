import torch
import torch.nn.functional as F


class LSTMCell(torch.nn.Module):
    """
    LSTMCell is a class that represents a single LSTM cell in a recurrent neural network.

    Args:
        input_size (int): The size of the input tensor.
        hidden_size (int): The size of the hidden state tensor.

    Attributes:
        weights (torch.nn.Linear): The linear layer that computes the weighted sum of the input and hidden state.
        hidden_size (int): The size of the hidden state tensor.

    Methods:
        init_weights(): Initializes the weights of the linear layer.
        forward(x, c, h): Performs the forward pass of the LSTM cell.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.weights = torch.nn.Linear(input_size + hidden_size, hidden_size * 4)
        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the linear layer using Xavier normal initialization
        and sets the biases to zero.
        """
        torch.nn.init.xavier_normal_(self.weights.weight)
        torch.nn.init.zeros_(self.weights.bias)

    def forward(self, x, c, h):
        """
        Performs the forward pass of the LSTM cell.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The cell state tensor.
            h (torch.Tensor): The hidden state tensor.

        Returns:
            h (torch.Tensor): The updated hidden state tensor.
            c (torch.Tensor): The updated cell state tensor.
        """
        cat = torch.concat((x, h), dim=1)
        gates = self.weights(cat)

        forget = F.sigmoid(gates[:, : self.hidden_size])
        input_gate = F.sigmoid(gates[:, self.hidden_size : 2 * self.hidden_size])
        output_gate = F.tanh(gates[:, 2 * self.hidden_size : 3 * self.hidden_size])
        cell_gate = F.sigmoid(gates[:, 3 * self.hidden_size :])

        c = forget * c + input_gate * cell_gate
        h = output_gate * F.tanh(c)

        return h, c


class GRUCell(torch.nn.Module):
    """
    GRUCell is a class that represents a single Gated Recurrent Unit (GRU) cell.

    Args:
        input_size (int): The number of expected features in the input `x`
        hidden_size (int): The number of features in the hidden state `h`

    Attributes:
        reset_x (torch.nn.Linear): Linear layer for the reset gate input weights
        reset_h (torch.nn.Linear): Linear layer for the reset gate hidden weights
        update_x (torch.nn.Linear): Linear layer for the update gate input weights
        update_h (torch.nn.Linear): Linear layer for the update gate hidden weights
        candidate_x (torch.nn.Linear): Linear layer for the candidate input weights
        candidate_h (torch.nn.Linear): Linear layer for the candidate hidden weights

    Methods:
        init_weights: Initializes the weights of the linear layers
        forward: Performs a forward pass of the GRU cell
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(GRUCell, self).__init__()
        self.reset_x = torch.nn.Linear(input_size, hidden_size)
        self.reset_h = torch.nn.Linear(hidden_size, hidden_size)
        self.update_x = torch.nn.Linear(input_size, hidden_size)
        self.update_h = torch.nn.Linear(hidden_size, hidden_size)
        self.candidate_x = torch.nn.Linear(input_size, hidden_size)
        self.candidate_h = torch.nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the linear layers using Xavier normal initialization
        and sets the biases to zero.
        """
        for weight in [
            self.reset_x,
            self.reset_h,
            self.update_x,
            self.update_h,
            self.candidate_x,
            self.candidate_h,
        ]:
            torch.nn.init.xavier_normal_(weight.weight)
            torch.nn.init.zeros_(weight.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the GRU cell.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size)
            h (torch.Tensor): The hidden state tensor of shape (batch_size, hidden_size)

        Returns:
            torch.Tensor: The updated hidden state tensor of shape (batch_size, hidden_size)
        """
        reset_gate = F.sigmoid(self.reset_x(x) + self.reset_h(h))
        update_gate = F.sigmoid(self.update_x(x) + self.update_h(h))
        candidate = F.tanh(self.update_x(x) + self.update_h(reset_gate * h))
        return update_gate * h + (1 - update_gate) * candidate
