import math

import torch

from src.model.input_embedding import InputEmbedding
from src.model.transformer_block import TransformerBlock


class Decoder(torch.nn.Module):
    _n_layers: int
    _input_dim: int
    _vocab_size: int
    _input_embedding: InputEmbedding
    _transformer_blocks: torch.nn.ModuleList
    _linear_final_layer: torch.nn.Linear
    _layer_norm: torch.nn.LayerNorm
    _gpt_2_like: bool

    def __init__(
        self,
        n_layers: int,
        max_length: int,
        input_dim: int,
        vocab_size: int,
        heads: int,
        dropout: float,
        gpt_2_like: bool = False,
    ) -> None:
        super().__init__()

        self._n_layers = n_layers
        self._input_dim = input_dim
        self._vocab_size = vocab_size
        self._input_embedding = InputEmbedding(
            max_length=max_length,
            input_dim=self._input_dim,
            vocab_size=self._vocab_size,
            dropout=dropout,
        )
        # GPT-2 scales the projections that write into the residual stream by
        # 1/sqrt(N), N being the number of residual additions (two per block).
        residual_scale = 1 / math.sqrt(2 * n_layers) if gpt_2_like else 1.0
        self._transformer_blocks = torch.nn.ModuleList(
            TransformerBlock(
                model_dim=self._input_dim,
                hidden_dim=self._input_dim * 4,
                heads=heads,
                max_length=max_length,
                dropout=dropout,
                gpt_2_like=gpt_2_like,
                residual_scale=residual_scale,
            )
            for _ in range(self._n_layers)
        )
        self._linear_final_layer = torch.nn.Linear(
            in_features=self._input_dim, out_features=self._vocab_size
        )
        self._layer_norm = torch.nn.LayerNorm(self._input_dim)
        self._gpt_2_like = gpt_2_like

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input_embedding(x)

        for block in self._transformer_blocks:
            x = block(x)

        if self._gpt_2_like:
            x = self._layer_norm(x)

        return self._linear_final_layer(x)
