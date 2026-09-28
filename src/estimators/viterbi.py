from random import Random
from typing import Final

import nltk

from src.data import language
from src.estimators.estimator import Estimator


class Viterbi(Estimator):
    sample_size: Final[int]
    seed: Final[int]

    def __init__(self, sample_size: int, seed: int = 0):
        self.sample_size = sample_size
        self.seed = seed

    def get_entropy(self, grammar: nltk.PCFG) -> float:
        rng = Random(self.seed)
        samples = [language.sample(grammar, rng) for _ in range(self.sample_size)]

        return sum(-log_prob for _, log_prob in samples) / sum(
            len(tokens) for tokens, _ in samples
        )
