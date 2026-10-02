import torch

from src.model.feedforward import FeedForward
from src.model.masked_self_attention import MaskedSelfAttention


class TransformerBlock(torch.nn.Module):
    _attention_norm: torch.nn.LayerNorm
    _feedforward_norm: torch.nn.LayerNorm
    _feedforward: FeedForward
    _multihead_attention: MaskedSelfAttention
    _gpt_2_like: bool

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        heads: int,
        max_length: int,
        dropout: float,
        gpt_2_like: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if model_dim % heads != 0:
            raise ValueError(
                f"model_dim {model_dim} is not divisible by {heads} heads."
            )

        self._attention_norm = torch.nn.LayerNorm(model_dim)
        self._feedforward_norm = torch.nn.LayerNorm(model_dim)
        self._feedforward = FeedForward(
            input_dim=model_dim, output_dim=hidden_dim, residual_scale=residual_scale
        )
        self._multihead_attention = MaskedSelfAttention(
            model_dim=model_dim,
            head_dim=model_dim // heads,
            heads=heads,
            max_length=max_length,
            dropout=dropout,
            masked=True,
            residual_scale=residual_scale,
        )
        self._gpt_2_like = gpt_2_like

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gpt_2_like:
            x = x + self._multihead_attention(self._attention_norm(x))
            x = x + self._feedforward(self._feedforward_norm(x))
        else:
            x = self._attention_norm(x + self._multihead_attention(x))
            x = self._feedforward_norm(x + self._feedforward(x))

        return x
