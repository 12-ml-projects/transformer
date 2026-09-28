import math
from itertools import pairwise

import nltk
import pytest

from src.data import language
from src.estimators.analytic import Analytic
from src.estimators.viterbi import Viterbi
from tests.conftest import LanguagePath


@pytest.fixture()
def language_1(language_spec: LanguagePath) -> nltk.grammar.PCFG:
    spec = language_spec("language_1")

    return language.load(spec)


@pytest.fixture(params=["bernoulli", "geometric"])
def ln_2_language(
    request: pytest.FixtureRequest, language_spec: LanguagePath
) -> nltk.grammar.PCFG:
    return language.load(language_spec(request.param))


class TestViterbi:
    def test_analytic_entropy_is_exact(self, ln_2_language: nltk.grammar.PCFG):
        assert Analytic().get_entropy(ln_2_language) == pytest.approx(math.log(2))

    def test_estimated_entropy_is_exact(self, ln_2_language: nltk.grammar.PCFG):
        entropy = Viterbi(1_000, seed=0).get_entropy(ln_2_language)

        assert entropy == pytest.approx(math.log(2))

    def test_estimation_converges(self, language_1: nltk.grammar.PCFG):
        entropies = [
            Viterbi(sample_size, seed=0).get_entropy(language_1)
            for sample_size in range(20_000, 100_000, 20_000)
        ]

        assert all(abs(h_1 - h_2) < 1e-3 for h_1, h_2 in pairwise(entropies))

    def test_correctness(self, language_1: nltk.grammar.PCFG):
        analytic = Analytic()

        assert (
            abs(
                Viterbi(40_000, seed=0).get_entropy(language_1)
                - analytic.get_entropy(language_1)
            )
            < 1e-3
        )
