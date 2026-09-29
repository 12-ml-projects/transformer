import math

import torch


class FeedForward(torch.nn.Module):
    _w_1: torch.nn.Parameter
    _b_1: torch.nn.Parameter
    _w_2: torch.nn.Parameter
    _b_2: torch.nn.Parameter

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        he_1: float = math.sqrt(2 / input_dim)
        he_2: float = math.sqrt(2 / output_dim)

        self._w_1 = torch.nn.Parameter(
            torch.randn(
                input_dim,
                output_dim,
            )
            * he_1,
        )
        self._b_1 = torch.nn.Parameter(
            torch.zeros(output_dim),
        )
        self._w_2 = torch.nn.Parameter(
            torch.randn(
                output_dim,
                input_dim,
            )
            * he_2,
        )
        self._b_2 = torch.nn.Parameter(
            torch.zeros(input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.gelu(x @ self._w_1 + self._b_1) @ self._w_2 + self._b_2
        )
