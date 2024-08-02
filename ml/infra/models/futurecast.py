import torch
import torch.nn.functional as F

from infra.models.rnn import LSTMCell, GRUCell
from infra.models.attention import Attention

from typing import Generator


class FuturecasterRegressor(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super(FuturecasterRegressor, self).__init__()
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


class Futurecaster(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        input_sequence_length: int,
        output_sequence_length: int,
        teacher_forcing: bool = False,
    ):
        super(Futurecaster, self).__init__()
        self.encoder_cell = LSTMCell(input_size, hidden_size)
        self.decoder_cell = LSTMCell(output_size, hidden_size)
        self.regressor = FuturecasterRegressor(hidden_size, output_size)

        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        self.teacher_forcing = teacher_forcing

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        c: torch.Tensor = None,
        h: torch.Tensor = None,
    ) -> torch.Tensor:
        if c is None:
            c = torch.zeros(x.size(0), self.encoder_cell.hidden_size).to(x.device)
        if h is None:
            h = torch.zeros(x.size(0), self.encoder_cell.hidden_size).to(x.device)

        encoder_hidden_states = []
        for i in range(self.input_sequence_length):
            h, c = self.encoder_cell(x[:, i], c, h)
            encoder_hidden_states.append(h)

        h, c = self.decoder_cell(x[:, -1, 0:1], c, h)
        closing_prices = [self.regressor(h)]

        if self.training and y is not None and self.teacher_forcing:
            for i in range(self.output_sequence_length - 1):
                h, c = self.decoder_cell(y[:, i : i + 1], c, h)
                closing_prices.append(self.regressor(h))
        else:
            for _ in range(self.output_sequence_length - 1):
                h, c = self.decoder_cell(closing_prices[-1], c, h)
                closing_prices.append(self.regressor(h))

        return torch.concat(closing_prices, dim=1)


class AttentiveFuturecaster(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        input_sequence_length: int,
        output_sequence_length: int,
        teacher_forcing_ratio: float | None = None,
    ):
        super(AttentiveFuturecaster, self).__init__()
        self.encoder = GRUCell(input_size, hidden_size)
        self.decoder = GRUCell(output_size, hidden_size)
        self.regressor = FuturecasterRegressor(hidden_size, output_size)
        self.attention = Attention(output_size, hidden_size, hidden_size)

        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        self.teacher_forcing = teacher_forcing_ratio is not None
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def _teacher_forcing_stream(
        self, x: torch.Tensor, y: torch.Tensor, closing_prices: list[torch.Tensor]
    ) -> Generator[torch.Tensor, None, None]:
        yield x[:, -1, 0:1]
        for i in range(self.output_sequence_length - 1):
            random = torch.rand(1).item()
            if random < self.teacher_forcing_ratio:
                # yield true label
                yield y[:, i : i + 1]
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

        closing_prices = [x[:, -1, 0:1]]
        stream = self._inference_stream(closing_prices)
        if self.training and y is not None and self.teacher_forcing:
            stream = self._teacher_forcing_stream(x, y, closing_prices)

        for prev_close in stream:
            attn_embedding = self.attention(
                prev_close.unsqueeze(1), hidden_states
            ).squeeze()
            h = self.decoder(prev_close, attn_embedding)
            closing_prices.append(self.regressor(h))

        return torch.concat(closing_prices[1:], dim=1)
