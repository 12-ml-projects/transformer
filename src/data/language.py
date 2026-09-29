import math
from functools import cache
from pathlib import Path
from random import Random
from typing import TypeAlias

import torch
from nltk import PCFG, Nonterminal

Nats: TypeAlias = float
Sentence: TypeAlias = list[str]


def load(path: Path) -> PCFG:
    return PCFG.fromstring(path.read_text())


@cache
def vocab_dict(grammar: PCFG) -> dict[str, int]:
    return {symbol: i for i, symbol in enumerate(vocabulary(grammar))}


@cache
def vocabulary(grammar: PCFG) -> list[str]:
    return sorted(
        {
            symbol
            for production in grammar.productions()
            for symbol in production.rhs()
            if not isinstance(symbol, Nonterminal)
        }
    )


def collate(batch: list[Sentence], grammar: PCFG) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a batch of token windows into (inputs, targets) id tensors."""
    ids = torch.tensor([to_vector(window, grammar) for window in batch])

    return ids[:, :-1], ids[:, 1:]


def to_vector(sentence: Sentence, grammar: PCFG) -> list[int]:
    v_dict = vocab_dict(grammar)

    return [v_dict[token] for token in sentence]


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
