import json
import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Iterable, TypeAlias

Document: TypeAlias = str
ID: TypeAlias = int
Word: TypeAlias = tuple[ID, ...]
Merges: TypeAlias = dict[tuple[ID, ID], ID]


PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


class BPE:
    """A byte-level BPE tokenizer defined by its ordered merges."""

    merges: Merges
    vocab: dict[ID, bytes]

    def __init__(self, merges: Merges):
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(merges.items(), key=lambda item: item[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, documents: Iterable[Document], n_merges: int) -> "BPE":
        return cls(modified_bpe(documents, n_merges))

    @classmethod
    def load(cls, path: Path) -> "BPE":
        pairs = json.loads(path.read_text())

        return cls({(a, b): 256 + i for i, (a, b) in enumerate(pairs)})

    def save(self, path: Path) -> None:
        pairs = sorted(self.merges, key=self.merges.__getitem__)
        path.write_text(json.dumps(pairs))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: Document) -> tuple[ID, ...]:
        pretokenized = pretokenize(text)
        words = []

        for word in pretokenized:
            encoded_word = tuple(word.encode("utf-8"))

            while True:
                pairs = pairwise(encoded_word)

                match = min(
                    pairs,
                    key=lambda pair: self.merges.get(pair, math.inf),
                    default=None,
                )
                if match is None or match not in self.merges:
                    break

                new_id = self.merges[match]
                encoded_word = merge(encoded_word, match, new_id)

            words.append(encoded_word)

        return tuple(id for word in words for id in word)

    def decode(self, ids: tuple[ID, ...]) -> Document:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")


def pretokenize(document: Document) -> list[str]:
    return re.findall(PATTERN, document)


def modified_bpe(documents: Iterable[Document], n_merges: int) -> Merges:
    merges: Merges = {}

    word_counts: Counter[str] = Counter()
    for document in documents:
        word_counts.update(pretokenize(document))

    words: dict[Word, int] = {
        tuple(word.encode("utf-8")): count for word, count in word_counts.items()
    }

    for _ in range(n_merges):
        pair_counts = count_pairs(words)
        if not pair_counts:
            break

        best = max(pair_counts, key=pair_counts.__getitem__)
        new_id = 256 + len(merges)
        merges[best] = new_id

        words = {merge(word, best, new_id): count for word, count in words.items()}

    return merges


def count_pairs(words: dict[Word, int]) -> Counter[tuple[ID, ID]]:
    """Count adjacent pairs within each word, weighted by the word's frequency."""
    pair_counts: Counter[tuple[ID, ID]] = Counter()

    for word, count in words.items():
        for pair in pairwise(word):
            pair_counts[pair] += count

    return pair_counts


def merge(word: Word, pair: tuple[ID, ID], new_id: ID) -> Word:
    """Replace every non-overlapping occurrence of pair in word, left to right."""
    merged: list[ID] = []
    i = 0

    while i < len(word):
        if word[i : i + 2] == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)
