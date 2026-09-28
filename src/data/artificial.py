from random import Random
from typing import Final, overload

import nltk
from torch.utils.data import Dataset

from src.data import language


class ArtificialDataset(Dataset[list[str]]):
    grammar: Final[nltk.PCFG]
    seed: Final[int]
    context_length: Final[int]
    _dataset_size: Final[int]
    _cache: dict[int, list[str]]

    def __init__(
        self,
        grammar: nltk.PCFG,
        seed: int = 0,
        dataset_size: int = 1000,
        context_length: int = 32,
    ):
        self.grammar = grammar
        self.seed = seed
        self.context_length = context_length
        self._dataset_size = dataset_size
        self._cache = {}

    @overload
    def __getitem__(self, index: int) -> list[str]: ...

    @overload
    def __getitem__(self, index: slice) -> list[list[str]]: ...

    def __getitem__(self, index: int | slice) -> list[str] | list[list[str]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if not 0 <= index < self._dataset_size:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self._dataset_size}"
            )

        if index not in self._cache:
            rng = Random(f"{self.seed}:{index}")
            tokens: list[str] = []
            while len(tokens) < self.context_length + 1:
                sentence, _ = language.sample(self.grammar, rng)
                tokens.extend(sentence)
            self._cache[index] = tokens[: self.context_length + 1]

        return self._cache[index]

    def __len__(self) -> int:
        return self._dataset_size
