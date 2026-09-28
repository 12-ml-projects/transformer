import math
from typing import TypeAlias

import nltk
import numpy as np

from src.estimators.estimator import Estimator

H: TypeAlias = np.ndarray
L: TypeAlias = np.ndarray


class Analytic(Estimator):
    def get_entropy(self, grammar: nltk.grammar.PCFG) -> float:
        categories = self.get_categories(grammar)
        h, length = self.get_quantities(grammar)
        start = categories.index(grammar.start())

        return float(h[start] / length[start])

    def get_categories(self, grammar: nltk.grammar.PCFG) -> list[nltk.Nonterminal]:
        return sorted({prod.lhs() for prod in grammar.productions()}, key=str)

    def get_matrix(self, grammar: nltk.grammar.PCFG) -> np.ndarray:
        categories = self.get_categories(grammar)
        matrix = np.zeros((len(categories), len(categories)))

        for row, col in np.ndindex(matrix.shape):
            prods = grammar.productions(lhs=categories[row])
            matrix[row, col] = sum(
                prod.prob() * prod.rhs().count(categories[col]) for prod in prods
            )

        return matrix

    def get_quantities(self, grammar: nltk.grammar.PCFG) -> tuple[H, L]:
        categories = self.get_categories(grammar)
        m = self.get_matrix(grammar)

        spectral_radius = max(abs(np.linalg.eigvals(m)))
        if spectral_radius >= 1:
            raise ValueError(
                f"Grammar is not consistent: spectral radius {spectral_radius} >= 1"
            )

        c_h = np.zeros(len(categories))
        c_l = np.zeros(len(categories))

        for idx, category in enumerate(categories):
            for prod in grammar.productions(lhs=category):
                p = prod.prob()
                terminals = sum(
                    not isinstance(symbol, nltk.Nonterminal) for symbol in prod.rhs()
                )
                c_h[idx] -= p * math.log(p)
                c_l[idx] += p * terminals

        identity = np.eye(len(categories))

        return np.linalg.solve(identity - m, c_h), np.linalg.solve(identity - m, c_l)
