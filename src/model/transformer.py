import torch

from src.model.input_embedding import InputEmbedding
from src.model.transformer_block import TransformerBlock


class Decoder(torch.nn.Module):
    _n_dec_layers: int
    _input_dim: int
    _vocab_size: int
    _input_embedding: InputEmbedding
    _transformer_blocks: torch.nn.ModuleList
    _linear_final_layer: torch.nn.Linear

    def __init__(
        self,
        n_dec_layers: int,
        max_length: int,
        input_dim: int,
        vocab_size: int,
        dropout: float,
    ):
        super().__init__()

        self._n_dec_layers = n_dec_layers
        self._input_dim = input_dim
        self._vocab_size = vocab_size
        self._input_embedding = InputEmbedding(
            max_length=max_length,
            input_dim=self._input_dim,
            vocab_size=self._vocab_size,
            dropout=dropout,
        )
        self._transformer_blocks = torch.nn.ModuleList(
            TransformerBlock(model_dim=self._input_dim, hidden_dim=self._input_dim * 4)
            for _ in range(self._n_dec_layers)
        )
        self._linear_final_layer = torch.nn.Linear(
            in_features=self._input_dim, out_features=self._vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input_embedding(x)

        for block in self._transformer_blocks:
            x = block(x)

        return self._linear_final_layer(x)
