from pathlib import Path
from typing import Protocol

import pytest

from src.data import language
from src.data.artificial import ArtificialDataset

CURRENT_FILE = Path(__file__)
LANGUAGE_SPECS = CURRENT_FILE.parent / "language_specs"
assert LANGUAGE_SPECS.exists()


class LanguagePath(Protocol):
    def __call__(self, language_name: str) -> Path: ...


class DatasetFactory(Protocol):
    def __call__(self, name: str, seed: int = 0) -> ArtificialDataset: ...


@pytest.fixture()
def language_spec() -> LanguagePath:
    def _language_spec(language_name: str) -> Path:
        path = LANGUAGE_SPECS / f"{language_name}.txt"

        if not path.exists():
            raise FileNotFoundError(f"Language file not found: {path}")

        return path

    return _language_spec


@pytest.fixture()
def dataset(language_spec: LanguagePath) -> DatasetFactory:
    def _artificial_dataset(name: str, seed: int = 0) -> ArtificialDataset:
        grammar = language.load(language_spec(name))

        return ArtificialDataset(grammar, seed=seed)

    return _artificial_dataset
