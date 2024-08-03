from typing import Generator

import torch
import torch.nn.functional as F

from infra.models.attention import Attention
from infra.models.rnn import GRUCell


class WalkerRegressor(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super(WalkerRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, output_size)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DiscreteWalker(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        input_sequence_length: int,
        output_sequence_length: int,
        teacher_forcing_ratio: float | None = None,
    ):
        super(DiscreteWalker, self).__init__()
        self.encoder = GRUCell(input_size, hidden_size)
        self.decoder = GRUCell(output_size + hidden_size, hidden_size)
        self.regressor = WalkerRegressor(hidden_size, output_size)
        self.attention = Attention(output_size, hidden_size, hidden_size)

        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.output_size = output_size

        self.teacher_forcing = teacher_forcing_ratio is not None
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def _teacher_forcing_stream(
        self, x: torch.Tensor, y: torch.Tensor, closing_prices: list[torch.Tensor]
    ) -> Generator[torch.Tensor, None, None]:
        yield x[:, -1, : self.output_size]
        for i in range(self.output_sequence_length - 1):
            random = torch.rand(1).item()
            if random < self.teacher_forcing_ratio:
                # yield true label
                yield y[:, i]
            else:
                # yield the model's prediction
                yield closing_prices[-1]

    def _inference_stream(
        self, closing_prices: list[torch.Tensor]
    ) -> Generator[torch.Tensor, None, None]:
        for _ in range(self.output_sequence_length):
            yield closing_prices[-1]

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        h: torch.Tensor = None,
    ) -> torch.Tensor:
        # TODO: consider attending over the output hidden states of the decoder
        if h is None:
            h = torch.randn(x.size(0), self.encoder.hidden_size).to(x.device)

        hidden_states = []
        for i in range(self.input_sequence_length):
            h = self.encoder(x[:, i], h)
            hidden_states.append(h)

        hidden_states = torch.stack(hidden_states, dim=1)

        walk_deltas = [x[:, -1, : self.output_size]]
        stream = self._inference_stream(walk_deltas)
        if self.training and y is not None and self.teacher_forcing:
            stream = self._teacher_forcing_stream(x, y, walk_deltas)

        h = torch.randn(x.size(0), self.encoder.hidden_size).to(x.device)
        for prev_close in stream:
            attn_embedding = self.attention(prev_close.unsqueeze(1), hidden_states)[
                :, 0
            ]
            h = self.decoder(torch.concat([prev_close, attn_embedding], dim=1), h)
            walk_deltas.append(self.regressor(h))

        return torch.stack(walk_deltas[1:], dim=1)
