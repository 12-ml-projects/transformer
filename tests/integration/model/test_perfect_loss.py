import math
from functools import partial

import pytest
from torch.utils.data import DataLoader

from src.data.language import collate
from src.estimators.analytic import Analytic
from src.experiment.training_loop import train
from tests.conftest import DatasetFactory
from tests.integration.model.conftest import SmallModelFactory


class TestOverfit:
    def test_can_reach_0_loss_if_possible(
        self, dataset: DatasetFactory, small_model: SmallModelFactory
    ):
        """
        Expect the model to drop to near zero training loss if tested on
        repeating training data ["test_1", "test_2", ..., "test_1", ...].
        """
        ds, pcfg = dataset("constant", context_length=32, dataset_size=100)
        model, optimizer, loss_fn, device = small_model(pcfg)

        max_epochs = 25

        collator = partial(collate, grammar=pcfg)

        for _, valid_loss in train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=DataLoader(ds, batch_size=4, collate_fn=collator),
            valid_loader=DataLoader(ds, batch_size=4, collate_fn=collator),
            n_epochs=max_epochs,
            device=device,
        ):
            if valid_loss < 0.001:
                return

        pytest.fail("Model did not learn the training pattern")

    def test_can_reach_theoretical_best_loss(
        self, dataset: DatasetFactory, small_model: SmallModelFactory
    ):
        """
        Expect the validation loss to approach, but never beat, the entropy limit on
        "a x b | c x d". Predicting b or d after x requires attending two tokens back.
        """
        context_length = 32
        train_ds, pcfg = dataset(
            "context", seed=0, context_length=context_length, dataset_size=1_000
        )
        valid_ds, _ = dataset(
            "context", seed=1, context_length=context_length, dataset_size=1_000
        )
        model, optimizer, loss_fn, device = small_model(pcfg)

        # The first token of each window is only ever an input, never a target, and
        # here it is exactly the random a/c choice. The targets therefore carry
        # (h * (S + 1) - H_first) / S nats per token rather than h: 10 ln 2 / 32
        # instead of ln 2 / 3 for S = 32. A perfect model reaches this lower value.
        entropy_first_token = math.log(2)
        limit = (
            Analytic().get_entropy(pcfg) * (context_length + 1) - entropy_first_token
        ) / context_length

        collator = partial(collate, grammar=pcfg)

        for _, valid_loss in train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=DataLoader(
                train_ds, batch_size=32, shuffle=True, collate_fn=collator
            ),
            valid_loader=DataLoader(valid_ds, batch_size=32, collate_fn=collator),
            n_epochs=25,
            device=device,
        ):
            assert valid_loss > limit - 1e-3, "Validation loss beat the entropy limit"

            if valid_loss < limit + 1e-2:
                return

        pytest.fail(f"Validation loss did not approach the entropy limit {limit:.4f}")
