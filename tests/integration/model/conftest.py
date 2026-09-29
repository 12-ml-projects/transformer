from typing import Protocol

import pytest
import torch
from nltk import PCFG

from src.data import language
from src.model.transformer import Decoder


class SmallModelFactory(Protocol):
    def __call__(
        self, pcfg: PCFG
    ) -> tuple[Decoder, torch.optim.Optimizer, torch.nn.Module, torch.device]: ...


@pytest.fixture()
def small_model() -> SmallModelFactory:
    def _small_model(
        pcfg: PCFG,
    ) -> tuple[Decoder, torch.optim.Optimizer, torch.nn.Module, torch.device]:
        torch.manual_seed(0)
        device = torch.device("cpu")

        decoder = Decoder(
            n_layers=2,
            max_length=32,
            input_dim=64,
            vocab_size=len(language.vocabulary(pcfg)),
            heads=4,
            dropout=0.0,
        ).to(device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

        return decoder, optimizer, torch.nn.CrossEntropyLoss(), device

    return _small_model
