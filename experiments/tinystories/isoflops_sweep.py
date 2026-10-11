"""Run an IsoFLOPs sweep: for each budget, train a few depths around the size that
Chinchilla's rule of thumb predicts, skipping runs this sweep has already done."""

import math
from functools import cache

import click
import mlflow
import numpy as np

from experiments.tinystories.isoflops_run import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EXPERIMENT_NAME,
    count_compute_params,
    count_non_embedding_params,
    model_config,
    run,
    tokens_for_budget,
)
from src.model.transformer import Decoder

MAX_LAYERS = 24  # d_model 768: far beyond what a laptop budget can train


@click.command()
@click.option(
    "--budget",
    "-b",
    "budgets",
    type=float,
    multiple=True,
    required=True,
    help="Compute budget in FLOPs; repeat for several budgets.",
)
@click.option("--n-sizes", default=5, help="Number of depths to train per budget.")
@click.option("--sweep", default="v1", help="Tag grouping the runs of one sweep.")
@click.option("--mlflow-tracking-uri", default="http://localhost:8888")
@click.option("--dry-run", is_flag=True, help="Only print the plan and its duration.")
def main(
    budgets: tuple[float, ...],
    n_sizes: int,
    sweep: str,
    mlflow_tracking_uri: str,
    dry_run: bool,
) -> None:
    """Train every planned (budget, depth) pair that this sweep has not done yet."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    done = set() if dry_run else finished_runs(sweep)
    plan = [
        (budget, n_layers)
        for budget in sorted(budgets)
        for n_layers in depths_around_optimum(budget, n_sizes)
        if (budget, n_layers) not in done
    ]

    click.echo(f"{len(plan)} runs to do")
    for budget, n_layers in plan:
        model = make_model(n_layers)
        compute_params = count_compute_params(model)
        n_tokens = tokens_for_budget(budget, compute_params)
        click.echo(
            f"  C={budget:.0e}  L={n_layers:>2}  "
            f"N={count_non_embedding_params(model):>10,}  D={n_tokens:>12,}  "
            f"FLOPs={6 * compute_params * n_tokens:>12,}"
        )

    if dry_run:
        return

    for budget, n_layers in plan:
        with mlflow.start_run(
            run_name=f"C={budget:.0e}-L={n_layers}", tags={"sweep": sweep}
        ):
            result = run(budget, n_layers)
        click.echo(
            f"C={budget:.0e} L={n_layers}: valid loss {result['valid_loss']:.4f}"
        )


def depths_around_optimum(budget: float, n_sizes: int) -> list[int]:
    """Depths whose sizes spread log-evenly from a third to three times the size
    Chinchilla's ~20 tokens per parameter predicts (C = 6 N D with D = 20 N, so
    N = sqrt(C/120)), keeping only those that train on at least two batches.

    Width grows with depth, so N grows roughly with the cube of the depth: each
    target size is matched to the depth whose size is closest on a log scale. The
    prediction counts the parameters that cost compute, output layer included."""
    optimum = math.sqrt(budget / 120)
    targets = np.geomspace(optimum / 3, optimum * 3, n_sizes)
    depths = range(1, MAX_LAYERS + 1)

    def closest_depth(target: float) -> int:
        return min(depths, key=lambda n: abs(math.log(compute_params(n) / target)))

    return sorted(
        {
            n_layers
            for n_layers in map(closest_depth, targets)
            if tokens_for_budget(budget, compute_params(n_layers))
            >= 2 * BATCH_SIZE * CONTEXT_LENGTH
        }
    )


def make_model(n_layers: int) -> Decoder:
    return Decoder(**model_config(n_layers))


@cache
def compute_params(n_layers: int) -> int:
    return count_compute_params(make_model(n_layers))


def finished_runs(sweep: str) -> set[tuple[float, int]]:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.sweep = '{sweep}' and attributes.status = 'FINISHED'",
        output_format="list",
    )

    return {
        (float(r.data.params["budget"]), int(r.data.params["n_layers"])) for r in runs
    }


if __name__ == "__main__":
    main()
