from collections.abc import Iterable, Iterator
from typing import TypeAlias

import torch
from torch.optim.lr_scheduler import LRScheduler

Inputs: TypeAlias = torch.Tensor
Targets: TypeAlias = torch.Tensor
Batch: TypeAlias = tuple[Inputs, Targets]

TrainLoss: TypeAlias = float
ValidLoss: TypeAlias = float


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_loader: Iterable[Batch],
    valid_loader: Iterable[Batch],
    n_tokens: int,
    eval_every: int,
    device: torch.device,
    scheduler: LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> Iterator[tuple[TrainLoss, ValidLoss]]:
    """Train on n_tokens target tokens, passing over the training data as often as
    needed, and yield (train_loss, valid_loss) every eval_every tokens and at the end.

    The train loss is the average over the tokens since the previous evaluation.
    A scheduler, if given, is stepped once per optimizer step. With max_grad_norm,
    the global norm of all gradients is clipped to it before each step.
    """
    model.train()
    tokens_seen = 0
    next_eval = eval_every
    total_loss = 0.0
    n_loss_tokens = 0

    for inputs, targets in _endless(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = _loss(model, loss_fn, inputs, targets)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_tokens = targets.numel()
        total_loss += loss.item() * batch_tokens
        n_loss_tokens += batch_tokens
        tokens_seen += batch_tokens

        finished = tokens_seen >= n_tokens
        if tokens_seen >= next_eval or finished:
            yield total_loss / n_loss_tokens, evaluate(
                model, loss_fn, valid_loader, device
            )

            model.train()
            total_loss = 0.0
            n_loss_tokens = 0
            while next_eval <= tokens_seen:
                next_eval += eval_every

        if finished:
            return


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
) -> ValidLoss:
    model.eval()
    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            loss = _loss(model, loss_fn, inputs, targets)

            total_loss += loss.item() * targets.numel()
            n_tokens += targets.numel()

    return total_loss / n_tokens


def _endless(loader: Iterable[Batch]) -> Iterator[Batch]:
    """Pass over the loader again and again, so a shuffling loader reshuffles
    every pass (unlike itertools.cycle, which replays the first pass)."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch

        if empty:
            raise ValueError("The training loader yields no batches")


def _loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)

    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
