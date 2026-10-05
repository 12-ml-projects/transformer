from pathlib import Path

import numpy as np
import pytest

from src.bpe import BPE
from src.data.tiny_stories import END_OF_TEXT, TinyStoriesDataset

STORIES = ["Once upon a time.", "The end."]
CONTEXT_LENGTH = 4


@pytest.fixture()
def dataset(tmp_path: Path) -> TinyStoriesDataset:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"{story}\n{END_OF_TEXT}\n" for story in STORIES))

    return TinyStoriesDataset(
        tokenizer=BPE({}),  # no merges: one token per byte
        corpus=corpus,
        token_file=tmp_path / "tokens.bin",
        context_length=CONTEXT_LENGTH,
    )


class TestTinyStoriesDataset:
    def test_requires_load(self, dataset: TinyStoriesDataset):
        with pytest.raises(RuntimeError):
            len(dataset)

    def test_token_file_contains_stories_separated_by_end_of_text(
        self, dataset: TinyStoriesDataset, tmp_path: Path
    ):
        dataset.load()
        tokens = np.fromfile(tmp_path / "tokens.bin", dtype=np.uint16)

        expected: list[int] = []
        for story in STORIES:
            expected.extend(story.encode("utf-8"))
            expected.append(dataset.end_of_text_id)

        assert tokens.tolist() == expected
        assert not (tmp_path / "tokens.tmp").exists()

    def test_windows_are_consecutive_and_shifted(
        self, dataset: TinyStoriesDataset, tmp_path: Path
    ):
        dataset.load()
        tokens = np.fromfile(tmp_path / "tokens.bin", dtype=np.uint16)

        for i in range(len(dataset)):
            example = dataset[i]
            start = i * (CONTEXT_LENGTH + 1)
            window = tokens[start : start + CONTEXT_LENGTH + 1].tolist()

            assert example.inputs.tolist() == window[:-1]
            assert example.targets.tolist() == window[1:]

    def test_length_and_bounds(self, dataset: TinyStoriesDataset):
        dataset.load()
        n_tokens = sum(len(story) + 1 for story in STORIES)

        assert len(dataset) == n_tokens // (CONTEXT_LENGTH + 1)
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_load_reuses_existing_token_file(
        self, dataset: TinyStoriesDataset, tmp_path: Path
    ):
        token_file = tmp_path / "tokens.bin"
        token_file.write_bytes(np.arange(10, dtype=np.uint16).tobytes())

        dataset.load()

        assert dataset[0].inputs.tolist() == [0, 1, 2, 3]
