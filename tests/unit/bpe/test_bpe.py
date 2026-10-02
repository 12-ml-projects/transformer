import pytest

from src.bpe.bpe import modified_bpe, pretokenize


class TestBPE:
    def test_modified_bpe(self):
        """Test the modified BPE merges for a simple document.

        Here 'He' = (He), 'Hel' = (H(el)) and 'Hell' = ((H(el))l) are merges.
        """
        b = "Hello world"
        documents = [b]

        merges = modified_bpe(documents, n_merges=3)

        assert list("Hello".encode("utf-8")) == [72, 101, 108, 108, 111]
        assert list(" world".encode("utf-8")) == [32, 119, 111, 114, 108, 100]

        assert merges == {
            (72, 101): 256,
            (256, 108): 257,
            (257, 108): 258,
        }

    def test_number_of_merges(self):
        b = "Hello world"
        documents = [b]

        merges = modified_bpe(documents, n_merges=3)

        assert len(merges.keys()) == 3


class TestPretokenise:
    @pytest.mark.parametrize(
        "document, expected",
        [
            ("Hello world", ["Hello", " world"]),
            ("Hello world world world", ["Hello", " world", " world", " world"]),
        ],
    )
    def test_splits_by_whitespace(self, document: str, expected: list[str]):
        tokens = pretokenize(document)

        assert tokens == expected

    @pytest.mark.parametrize(
        "document, expected",
        [
            ("Hello world!", ["Hello", " world", "!"]),
            ("Shouting! Shouting!", ["Shouting", "!", " Shouting", "!"]),
            ("So, ", ["So", ",", " "]),
            ("Hm...", ["Hm", "..."]),
        ],
    )
    def test_word_followed_by_character(self, document: str, expected: list[str]):
        tokens = pretokenize(document)

        assert tokens == expected

    @pytest.mark.parametrize(
        "document, expected",
        [("Test\nTest", ["Test", "\n", "Test"]), (" \t\n", [" \t\n"])],
    )
    def test_whitespace_chars(self, document: str, expected: list[str]):
        tokens = pretokenize(document)

        assert tokens == expected
