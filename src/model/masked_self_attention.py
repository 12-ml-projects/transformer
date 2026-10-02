import math

import torch


class MaskedSelfAttention(torch.nn.Module):
    _query_matrices: torch.nn.ModuleList
    _key_matrices: torch.nn.ModuleList
    _value_matrices: torch.nn.ModuleList
    _output: torch.nn.Linear
    _dropout: torch.nn.Dropout
    _mask: torch.Tensor
    _head_dim: int
    _heads: int
    _masked: bool

    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        heads: int,
        max_length: int,
        dropout: float,
        masked: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self._query_matrices = torch.nn.ModuleList(
            torch.nn.Linear(model_dim, head_dim, bias=False) for _ in range(heads)
        )
        self._key_matrices = torch.nn.ModuleList(
            torch.nn.Linear(model_dim, head_dim, bias=False) for _ in range(heads)
        )
        self._value_matrices = torch.nn.ModuleList(
            torch.nn.Linear(model_dim, head_dim, bias=False) for _ in range(heads)
        )
        self._output = torch.nn.Linear(head_dim * heads, model_dim, bias=False)
        with torch.no_grad():
            self._output.weight.mul_(residual_scale)
        self._dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer(
            "_mask", torch.tril(torch.ones(max_length, max_length)), persistent=False
        )

        self._head_dim = head_dim
        self._heads = heads
        self._masked = masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = [q(x) for q in self._query_matrices]
        k = [k(x) for k in self._key_matrices]
        v = [v(x) for v in self._value_matrices]

        multihead = torch.concat(
            [self._attention(q[h], k[h], v[h]) for h in range(self._heads)],
            dim=-1,
        )

        return self._output(multihead)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(self._head_dim)

        if self._masked:
            sequence_length = q.shape[-2]
            mask = self._mask[:sequence_length, :sequence_length]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        return self._dropout(torch.softmax(scores, dim=-1)) @ v
