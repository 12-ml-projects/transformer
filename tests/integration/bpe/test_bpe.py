from pathlib import Path

import pytest

from src.bpe import BPE

TRAINING_TEXT = "Hello world"


class TestBPE:
    def test_train_creates_merges(self):
        bpe = BPE.train([TRAINING_TEXT], n_merges=3)

        assert len(bpe.merges) == 3
        assert bpe.merges == {
            (72, 101): 256,
            (256, 108): 257,
            (257, 108): 258,
        }

    def test_encode(self):
        bpe = BPE.train([TRAINING_TEXT], n_merges=3)
        encoded = bpe.encode(TRAINING_TEXT)

        assert encoded == (258, 111, 32, 119, 111, 114, 108, 100)

    def test_encode_applies_earliest_merge_first(self):
        a, b, c = map(ord, "abc")
        bpe = BPE({(b, c): 256, (a, b): 257})

        assert bpe.encode("abc") == (a, 256)

    def test_vocab_contains_merges(self):
        bpe = BPE.train([TRAINING_TEXT], n_merges=3)
        subwords = bpe.vocab.values()

        assert all(
            (merge.encode("utf-8") in subwords) for merge in ["He", "Hel", "Hell"]
        )

    def test_saved_tokenizer_loads_and_encodes_identically(self, tmp_path: Path):
        text = "Hello there, world!"
        trained = BPE.train([TRAINING_TEXT], n_merges=3)
        trained.save(tmp_path / "merges.json")

        loaded = BPE.load(tmp_path / "merges.json")

        assert loaded.merges == trained.merges
        assert loaded.encode(text) == trained.encode(text)
        assert loaded.decode(loaded.encode(text)) == text

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world",
            "Hello there, world!",
            "  double  spaces\n\ttabs",
            "",
            "héllo wörld",
            "🙂",
        ],
    )
    def test_decode_inverts_encode(self, text: str):
        bpe = BPE.train([TRAINING_TEXT], n_merges=3)

        assert bpe.decode(bpe.encode(text)) == text
