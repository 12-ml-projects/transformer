import math
from pathlib import Path
from random import Random
from typing import TypeAlias

from nltk import PCFG, Nonterminal

Nats: TypeAlias = float
Sentence: TypeAlias = list[str]


def load(path: Path) -> PCFG:
    return PCFG.fromstring(path.read_text())


def sample(grammar: PCFG, rng: Random) -> tuple[Sentence, Nats]:
    """Sample a sentence and its log probability in nats."""
    return _expand(grammar, rng, grammar.start())


def _expand(
    grammar: PCFG, rng: Random, symbol: Nonterminal | str
) -> tuple[Sentence, Nats]:
    if not isinstance(symbol, Nonterminal):
        return [symbol], 0.0

    productions = list(grammar.productions(lhs=symbol))
    production = rng.choices(
        productions,
        weights=[p.prob() for p in productions],
    )[0]

    tokens: Sentence = []
    log_prob = math.log(production.prob())

    for child in production.rhs():
        child_tokens, child_log_prob = _expand(grammar, rng, child)
        tokens.extend(child_tokens)
        log_prob += child_log_prob

    return tokens, log_prob
