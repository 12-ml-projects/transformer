import torch

from src.model.feedforward import FeedForward
from src.model.masked_self_attention import MaskedSelfAttention


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        heads: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if model_dim % heads != 0:
            raise ValueError(
                f"model_dim {model_dim} is not divisible by {heads} heads."
            )

        self._attention_norm = torch.nn.LayerNorm(model_dim)
        self._feedforward_norm = torch.nn.LayerNorm(model_dim)
        self._feedforward = FeedForward(input_dim=model_dim, output_dim=hidden_dim)
        self._multihead_attention = MaskedSelfAttention(
            model_dim=model_dim,
            head_dim=model_dim // heads,
            heads=heads,
            max_length=max_length,
            dropout=dropout,
            masked=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._attention_norm(self._multihead_attention(x) + x)
        x = self._feedforward_norm(self._feedforward(x) + x)

        return x
