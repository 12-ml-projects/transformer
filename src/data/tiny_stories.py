from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.bpe.bpe import BPE

END_OF_TEXT = "<|endoftext|>"


class Example(NamedTuple):
    """A window of tokens, split into what the model sees and what it predicts."""

    inputs: torch.Tensor
    targets: torch.Tensor


class TinyStoriesDataset(Dataset[Example]):
    """Fixed-length windows over a corpus encoded once into a uint16 token file."""

    _tokenizer: BPE
    _corpus: Path
    _token_file: Path
    _context_length: int
    _tokens: np.memmap[Any, np.dtype[np.uint16]] | None

    def __init__(
        self, tokenizer: BPE, corpus: Path, token_file: Path, context_length: int
    ) -> None:
        super().__init__()

        self._tokenizer = tokenizer
        self._corpus = corpus
        self._token_file = token_file
        self._context_length = context_length
        self._tokens = None

    @property
    def end_of_text_id(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def vocab_size(self) -> int:
        """The model's vocabulary: the tokenizer's plus the end-of-text token."""
        return self._tokenizer.vocab_size + 1

    def load(self, skip_existing: bool = True) -> None:
        if not self._token_file.exists() or not skip_existing:
            self._write_tokens()

        self._tokens = np.memmap(self._token_file, dtype=np.uint16, mode="r")

    def _write_tokens(self) -> None:
        """Encode the corpus, writing to a temporary file for safe concurrent writes."""
        if self.vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError(f"A vocabulary of {self.vocab_size} does not fit uint16")

        temp_path = self._token_file.with_suffix(".tmp")
        end_of_text = (self.end_of_text_id,)

        with temp_path.open("wb") as file:
            for story in read_documents(self._corpus):
                ids = self._tokenizer.encode(story) + end_of_text
                np.array(ids, dtype=np.uint16).tofile(file)

        temp_path.replace(self._token_file)

    def __getitem__(self, idx: int) -> Example:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for {len(self)} windows")

        start = idx * (self._context_length + 1)
        window = self._loaded_tokens()[start : start + self._context_length + 1]
        ids = torch.from_numpy(window.astype(np.int64))

        return Example(inputs=ids[:-1], targets=ids[1:])

    def __len__(self) -> int:
        return len(self._loaded_tokens()) // (self._context_length + 1)

    def _loaded_tokens(self) -> np.memmap[Any, np.dtype[np.uint16]]:
        if self._tokens is None:
            raise RuntimeError("Call load() before using the dataset")

        return self._tokens


def read_documents(path: Path) -> Iterator[str]:
    """Stream the stories one by one, splitting on the end-of-text marker."""
    with path.open(encoding="utf-8") as file:
        story: list[str] = []

        for line in file:
            if line.strip() == END_OF_TEXT:
                yield "".join(story).strip()
                story = []
            else:
                story.append(line)
