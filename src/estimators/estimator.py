import math
from abc import ABC, abstractmethod

import nltk


class Estimator(ABC):
    @abstractmethod
    def get_entropy(self, grammar: nltk.PCFG) -> float:
        pass

    def get_perplexity(self, grammar: nltk.PCFG) -> float:
        return math.exp(self.get_entropy(grammar))
