import math
from random import Random

import nltk
import pytest

from src.data import language
from tests.conftest import LanguagePath


@pytest.fixture()
def grammar(language_spec: LanguagePath) -> nltk.PCFG:
    return language.load(language_spec("language_1"))


class TestLanguage:
    def test_sentences_end_with_single_dot(self, grammar: nltk.PCFG):
        sentences = [language.sample(grammar, Random(i))[0] for i in range(10)]

        assert all(
            sentence[-1] == "." and "." not in sentence[:-1] for sentence in sentences
        )

    def test_sentences_have_exactly_one_parse(self, grammar: nltk.PCFG):
        parser = nltk.ChartParser(grammar)
        sentences = [language.sample(grammar, Random(i))[0] for i in range(100)]

        assert all(len(list(parser.parse(sentence))) == 1 for sentence in sentences)

    def test_log_prob_matches_viterbi_parse(self, grammar: nltk.PCFG):
        parser = nltk.ViterbiParser(grammar)

        for i in range(100):
            sentence, log_prob = language.sample(grammar, Random(i))
            (tree,) = parser.parse(sentence)

            assert log_prob == pytest.approx(math.log(tree.prob()))
