from collections.abc import Iterable, Iterator
from typing import TypeAlias

import torch

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
    n_epochs: int,
    device: torch.device,
) -> Iterator[tuple[TrainLoss, ValidLoss]]:
    """Train for n_epochs, yielding (train_loss, valid_loss) after each epoch."""
    for _ in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            loss = _loss(model, loss_fn, inputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

        yield total_loss / n_samples, evaluate(model, loss_fn, valid_loader, device)


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
) -> ValidLoss:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            loss = _loss(model, loss_fn, inputs, targets)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

    return total_loss / n_samples


def _loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)

    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
