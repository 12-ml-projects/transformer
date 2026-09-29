"""Train a model on the basic language `language_1` from the tests."""

import math
from functools import partial
from pathlib import Path
from typing import Any

import click
import mlflow
import optuna
import torch
from nltk.grammar import PCFG
from torch.utils.data import DataLoader

from src.data import language
from src.data.artificial import ArtificialDataset
from src.estimators.analytic import Analytic
from src.experiment import evaluate, train
from src.experiment.utils import get_device
from src.model.transformer import Decoder

EXPERIMENT_NAME = "simple-language"
RESULTS_DIR = Path(__file__).parent
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"

CONTEXT_LENGTH = 32
BATCH_SIZE = 32

TRAIN_SEED = 0
VALID_SEED = 1
TEST_SEED = 2

device = get_device()
loss_fn = torch.nn.CrossEntropyLoss()


@click.command()
@click.option("--n-trials", default=50, help="Number of optimization trials.")
@click.option("--n-epochs", default=10, help="Number of training epochs.")
@click.option(
    "--language-spec",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the language specification file.",
)
@click.option(
    "--mlflow-tracking-uri",
    default="http://localhost:8888",
    help="MLflow tracking URI.",
)
@click.option(
    "--pruner",
    type=click.Choice(["hyperband", "median", "none"]),
    default="hyperband",
    help="Strategy for stopping unpromising trials early.",
)
def train_model(
    n_trials: int,
    n_epochs: int,
    language_spec: Path,
    mlflow_tracking_uri: str,
    pruner: str,
) -> None:
    pcfg = language.load(language_spec)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(
        study_name=EXPERIMENT_NAME,
        storage=f"sqlite:///{RESULTS_DIR / 'optuna.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=make_pruner(pruner, n_epochs),
    )

    with mlflow.start_run(
        run_name=f"study-{len(study.trials)}",
        description="Optimization study for simple language model",
    ):
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "language_spec": language_spec.name,
                "pruner": pruner,
            }
        )
        study.optimize(
            partial(objective, n_epochs=n_epochs, pcfg=pcfg), n_trials=n_trials
        )

        best_run_id = study.best_trial.user_attrs["mlflow_run_id"]
        mlflow.set_tag("best_run_id", best_run_id)
        mlflow.MlflowClient().set_tag(best_run_id, "best", "true")
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_valid_loss", study.best_value)

        test_loss = evaluate_best(study.best_params, pcfg)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.MlflowClient().log_metric(best_run_id, "test_loss", test_loss)

    entropy = Analytic().get_entropy(pcfg)
    click.echo(f"Best trial:       {study.best_trial.number} {study.best_params}")
    click.echo(f"Valid loss:       {study.best_value:.4f}")
    click.echo(f"Test loss:        {test_loss:.4f}")
    click.echo(f"Entropy (floor):  {entropy:.4f} nats/token")


def evaluate_best(params: dict[str, Any], pcfg: PCFG) -> float:
    model = make_model(params, pcfg)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    return evaluate(model, loss_fn, make_loader(pcfg, seed=TEST_SEED), device)


def make_pruner(name: str, n_epochs: int) -> optuna.pruners.BasePruner:
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_epochs)
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=min(3, n_epochs)
        )

    return optuna.pruners.NopPruner()


def make_model(params: dict[str, Any], pcfg: PCFG) -> Decoder:
    return Decoder(
        n_layers=params["num_layers"],
        max_length=CONTEXT_LENGTH,
        input_dim=params["input_dim"],
        vocab_size=len(language.vocabulary(pcfg)),
        heads=params["heads"],
        dropout=params["dropout"],
    ).to(device)


def make_loader(pcfg: PCFG, seed: int, shuffle: bool = False) -> DataLoader[Any]:
    dataset = ArtificialDataset(pcfg, seed=seed, context_length=CONTEXT_LENGTH)
    collator = partial(language.collate, grammar=pcfg)

    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collator
    )


def objective(trial: optuna.trial.Trial, n_epochs: int, pcfg: PCFG) -> float:
    torch.manual_seed(trial.number)

    trial.suggest_int("num_layers", 1, 6)
    trial.suggest_float("dropout", 0.0, 0.2)
    trial.suggest_categorical("input_dim", [32, 64, 128, 256])
    trial.suggest_categorical("heads", [1, 2, 4, 8])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)

    model = make_model(trial.params, pcfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = make_loader(pcfg, seed=TRAIN_SEED, shuffle=True)
    valid_loader = make_loader(pcfg, seed=VALID_SEED)

    valid_loss = math.inf
    pruned = False

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        mlflow.log_params(
            trial.params
            | {
                "context_length": CONTEXT_LENGTH,
                "batch_size": BATCH_SIZE,
                "n_epochs": n_epochs,
            }
        )

        for epoch, (train_loss, valid_loss) in enumerate(
            train(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_loader=train_loader,
                valid_loader=valid_loader,
                n_epochs=n_epochs,
                device=device,
            )
        ):
            mlflow.log_metrics(
                {"train_loss": train_loss, "valid_loss": valid_loss}, step=epoch
            )

            trial.report(valid_loss, epoch)
            if trial.should_prune():
                pruned = True
                break

        mlflow.set_tag("pruned", pruned)

    if pruned:
        raise optuna.TrialPruned()

    if is_new_best(trial.study, valid_loss):
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    return valid_loss


def is_new_best(study: optuna.Study, valid_loss: float) -> bool:
    try:
        return valid_loss < study.best_value
    except ValueError:  # no completed trials yet
        return True


if __name__ == "__main__":
    train_model()
