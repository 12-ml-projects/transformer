import torch

from src.model.feedforward import FeedForward
from src.model.masked_self_attention import MaskedSelfAttention


class TransformerBlock(torch.nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self._layer_norm = torch.nn.LayerNorm(model_dim)
        self._feedforward = FeedForward(input_dim=model_dim, output_dim=hidden_dim)
        self._multihead_attention = MaskedSelfAttention(
            model_dim=model_dim,
            head_dim=model_dim // 8,
            heads=8,
            max_length=512,
            dropout=0.1,
            masked=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._multihead_attention(x) + x
        x = self._layer_norm(x)
        x = self._feedforward(x) + x
        x = self._layer_norm(x)

        return x
