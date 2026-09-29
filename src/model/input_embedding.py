import math
from typing import Final

import torch


class InputEmbedding(torch.nn.Module):
    _input_dim: Final[int]
    _vocab_size: Final[int]
    _embedding: torch.nn.Embedding
    _dropout: torch.nn.Dropout
    _pe: torch.Tensor

    def __init__(
        self, max_length: int, input_dim: int, vocab_size: int, dropout: float
    ):
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError("input_dim must be an even number.")

        self._input_dim = input_dim
        self._vocab_size = vocab_size
        self._embedding = torch.nn.Embedding(
            embedding_dim=self._input_dim,
            num_embeddings=self._vocab_size,
        )
        self._dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer("_pe", self._make_pe(max_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]

        if sequence_length > self._pe.size(0):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max length "
                f"{self._pe.size(0)}."
            )

        return self._dropout(self._embedding(x) + self._pe[:sequence_length])

    def _make_pe(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self._input_dim)

        positions = torch.arange(0, length, dtype=torch.float32).unsqueeze(-1)

        trig_args = torch.exp(
            torch.arange(0, self._input_dim, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / self._input_dim)
        )
        pe[:, 0::2] = torch.sin(positions * trig_args)
        pe[:, 1::2] = torch.cos(positions * trig_args)

        return pe
