import torch
import torch.nn.functional as F

from math import sqrt


class Attention(torch.nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int):
        super(Attention, self).__init__()
        self.key_matrix = torch.nn.Parameter(torch.zeros(key_dim, query_dim))
        self.value_matrix = torch.nn.Parameter(torch.zeros(key_dim, value_dim))
        self.query_dim = query_dim

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.key_matrix)
        torch.nn.init.xavier_normal_(self.value_matrix)

    def forward(self, query: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        key_vectors = torch.matmul(inputs, self.key_matrix)
        value_vectors = torch.matmul(inputs, self.value_matrix)

        scores = torch.bmm(query, key_vectors.transpose(1, 2)) / sqrt(self.query_dim)
        weights = torch.nn.functional.softmax(scores, dim=2)
        return torch.bmm(weights, value_vectors)
