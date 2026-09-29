import pytest

from tests.conftest import DatasetFactory


class TestArtificialLanguage:
    def test_iter_is_pure(self, dataset: DatasetFactory):
        language, _ = dataset("language_1", seed=0)
        sentence = language[10]

        assert sentence == language[10]

    def test_throws_error_if_index_out_of_range(self, dataset: DatasetFactory):
        language, _ = dataset("language_1", seed=0)

        with pytest.raises(IndexError):
            language[len(language)]

        with pytest.raises(IndexError):
            language[-1]

    def test_same_seed_gives_same_sentences(self, dataset: DatasetFactory):
        first, _ = dataset("language_1", seed=10)
        second, _ = dataset("language_1", seed=10)

        assert first[:10] == second[:10]

    def test_different_seeds_give_different_sentences(self, dataset: DatasetFactory):
        first, _ = dataset("language_1", seed=0)
        second, _ = dataset("language_1", seed=1)

        assert first[:10] != second[:10]

    def test_indices_give_unique_sentences(self, dataset: DatasetFactory):
        language, _ = dataset("language_1", seed=0)
        sentences = {tuple(sentence) for sentence in language[:10]}

        assert len(sentences) == 10

    def test_windows_have_context_length_plus_one_tokens(self, dataset: DatasetFactory):
        language, _ = dataset("language_1", seed=0)

        assert all(
            len(window) == language.context_length + 1 for window in language[:10]
        )
