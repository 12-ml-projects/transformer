import pytest
import torch

from src.model.input_embedding import InputEmbedding


class TestInputEmbedding:
    def test_shape_is_correct(self):
        embedding = get_embedding()

        x = get_x(2, 10)

        assert embedding(x).shape == (2, 10, 100)

    def test_throws_error_if_input_too_long(self):
        embedding = get_embedding(max_length=10)

        x = get_x(2, 11)

        with pytest.raises(
            ValueError, match="Sequence length 11 exceeds maximum length 10"
        ):
            embedding(x)

    def test_batch_independence(self):
        embedding = get_embedding(max_length=10, dropout=0.0).eval()

        x = get_x(2, 10)

        y = embedding(x)
        z = embedding(x[0].unsqueeze(0))

        assert torch.allclose(y[0], z[0], atol=1e-6)


def get_embedding(
    max_length: int = 10,
    input_dim: int = 100,
    vocab_size: int = 50,
    dropout: float = 0.1,
) -> InputEmbedding:
    return InputEmbedding(
        max_length=max_length,
        input_dim=input_dim,
        vocab_size=vocab_size,
        dropout=dropout,
    )


def get_x(batch_size: int, sequence_length: int, max: int = 10) -> torch.Tensor:
    return torch.randint(0, max, (batch_size, sequence_length))
