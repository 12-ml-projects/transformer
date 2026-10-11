import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1
) -> float:
    """Learning-rate multiplier: linear warmup to 1, then cosine decay to min_ratio
    at total_steps (GPT-3 / Chinchilla), staying at min_ratio afterwards."""
    # during warmup the learning rate increases linearly from 0 to 1
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    # progress starts at 0, up to 1; capped
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))

    # apply cosine decay to the learning rate, scaling it between 1 and min_ratio
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> LambdaLR:
    """Scale the optimizer's learning rate by cosine_with_warmup at every step."""
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"Need 0 <= warmup_steps < total_steps, got {warmup_steps}, {total_steps}"
        )

    return LambdaLR(
        optimizer,
        partial(
            cosine_with_warmup,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_ratio=min_ratio,
        ),
    )
