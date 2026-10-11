"""Plot the IsoFLOP profiles of a sweep and the scaling law through their minima.

Each budget gets a panel with the final validation loss of each model size and the
parabola fitted through it. The minima of those parabolas, one per budget, lie on
a line in log-log space: the scaling law N_opt = k C^a.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, LogLocator, NullFormatter
from numpy.polynomial import Polynomial

from experiments.tinystories.isoflops_run import EXPERIMENT_NAME

PLOTS = Path(__file__).parents[2] / "plots"
SIZE_LABELS = {
    "compute_params": "parameters costing compute per token",
    "n_params": "non-embedding parameters",
}
COLUMNS = 3

Budget = TypeVar("Budget", float, np.ndarray)  # one budget or an array of them


@dataclass(frozen=True)
class Optimum:
    """The minimum of one budget's parabola: the best size and its predicted loss."""

    budget: float
    size: float
    loss: float


@dataclass(frozen=True)
class ScalingLaw:
    """The compute-optimal model size as a power of the budget: N_opt = k C^a."""

    coefficient: float
    exponent: float

    def optimal_size(self, budget: Budget) -> Budget:
        return self.coefficient * budget**self.exponent


@click.command()
@click.option("--sweep", default="v1", help="Tag of the sweep to plot.")
@click.option(
    "--size",
    type=click.Choice(list(SIZE_LABELS)),
    default="compute_params",
    help="Which parameter count to use as model size.",
)
@click.option("--mlflow-tracking-uri", default="http://localhost:8888")
def main(sweep: str, size: str, mlflow_tracking_uri: str) -> None:
    """Plot the IsoFLOP profiles and the scaling law of a sweep into plots/."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    runs = fetch_runs(sweep)
    if runs.empty:
        raise click.ClickException(f"No finished runs for sweep '{sweep}'")

    save(plot_profiles(runs, size), f"isoflops_{sweep}.png")

    optima = find_optima(runs, size)
    law = fit_scaling_law(optima)
    if law is not None:
        save(plot_scaling_law(optima, law, size), f"isoflops_{sweep}_law.png")

    report(optima, law, size)


def fetch_runs(sweep: str) -> pd.DataFrame:
    """The finished runs of a sweep: budget, model sizes and final validation loss."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.sweep = '{sweep}' and attributes.status = 'FINISHED'",
    )

    return pd.DataFrame(
        {
            "budget": runs["params.budget"].astype(float),
            "n_params": runs["params.n_params"].astype(int),
            "compute_params": runs["params.compute_params"].astype(int),
            "valid_loss": runs["metrics.valid_loss"],
        }
    )


def find_optima(runs: pd.DataFrame, size: str) -> list[Optimum]:
    """The minimum of each budget's parabola, for budgets with at least three runs."""
    optima = []
    for budget in sorted(runs["budget"].unique()):
        budget_runs = runs[runs["budget"] == budget]
        parabola = fit_parabola(
            budget_runs[size].to_numpy(), budget_runs["valid_loss"].to_numpy()
        )
        if parabola is None:
            continue

        log_optimum = vertex(parabola)
        optima.append(
            Optimum(
                budget=float(budget),
                size=math.exp(log_optimum),
                loss=float(parabola(log_optimum)),
            )
        )

    return optima


def fit_scaling_law(optima: list[Optimum]) -> ScalingLaw | None:
    """A straight line through the optima in log-log space; None below two budgets."""
    if len(optima) < 2:
        return None

    budgets = np.log([optimum.budget for optimum in optima])
    sizes = np.log([optimum.size for optimum in optima])
    intercept, slope = Polynomial.fit(budgets, sizes, 1).convert().coef

    return ScalingLaw(coefficient=math.exp(intercept), exponent=float(slope))


def plot_profiles(runs: pd.DataFrame, size: str) -> Figure:
    """A grid with one IsoFLOP profile per budget, smallest budget first."""
    budgets = sorted(runs["budget"].unique())
    rows = math.ceil(len(budgets) / COLUMNS)
    figure, axes = plt.subplots(
        rows, COLUMNS, figsize=(4 * COLUMNS, 3.5 * rows), squeeze=False
    )

    for ax, budget in zip(axes.flat, budgets):
        plot_profile(ax, runs[runs["budget"] == budget], size)

    for ax in axes.flat[len(budgets) :]:
        ax.set_visible(False)

    figure.supxlabel(SIZE_LABELS[size])
    figure.supylabel("final validation loss (nats/token)")
    figure.tight_layout()

    return figure


def plot_profile(ax: Axes, runs: pd.DataFrame, size: str) -> None:
    """The runs of one budget, with the fitted parabola and its minimum."""
    budget = runs["budget"].iloc[0]
    sizes = runs[size].to_numpy()
    losses = runs["valid_loss"].to_numpy()

    ax.scatter(sizes, losses, zorder=3)
    format_log_axis(ax.xaxis)
    ax.set_xscale("log")
    ax.set_title(f"C = {budget:.2e} FLOPs")

    if size == "compute_params":
        ax.axvline(
            chinchilla_optimum(budget),
            color="tab:orange",
            linestyle="--",
            label="Chinchilla",
        )

    parabola = fit_parabola(sizes, losses)
    if parabola is not None:
        curve = np.geomspace(sizes.min(), sizes.max(), 200)
        ax.plot(curve, parabola(np.log(curve)), color="gray")

        optimum = math.exp(vertex(parabola))
        ax.axvline(optimum, color="gray", linestyle=":", label="fitted")
        ax.set_title(f"{ax.get_title()}\noptimum ≈ {optimum:,.0f}")

    ax.legend(loc="upper center", fontsize="small")


def plot_scaling_law(optima: list[Optimum], law: ScalingLaw, size: str) -> Figure:
    """The optimum of each budget with the fitted power law through them."""
    figure, ax = plt.subplots(figsize=(6, 4.5))
    measured = [optimum.budget for optimum in optima]
    budgets = np.geomspace(min(measured), max(measured), 100)

    ax.scatter(measured, [optimum.size for optimum in optima], zorder=3, label="optima")
    ax.plot(
        budgets,
        law.optimal_size(budgets),
        color="gray",
        label=f"fit: N ∝ C^{law.exponent:.2f}",
    )
    if size == "compute_params":
        ax.plot(
            budgets,
            [chinchilla_optimum(b) for b in budgets],
            color="tab:orange",
            linestyle="--",
            label="Chinchilla: N = √(C / 120)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    format_log_axis(ax.yaxis)
    ax.set_xlabel("compute budget C (FLOPs)")
    ax.set_ylabel(f"optimal size ({SIZE_LABELS[size]})")
    ax.legend()
    figure.tight_layout()

    return figure


def report(optima: list[Optimum], law: ScalingLaw | None, size: str) -> None:
    """Print the optimum per budget and the scaling law through them."""
    click.echo(f"\nOptimal sizes in {SIZE_LABELS[size]}:")
    for optimum in optima:
        line = f"  C = {optimum.budget:.2e}   N_opt = {optimum.size:>12,.0f}"
        if size == "compute_params":
            tokens = optimum.budget / (6 * optimum.size)
            line += f"   D_opt = {tokens:>14,.0f}   D/N = {tokens / optimum.size:5.1f}"
        click.echo(f"{line}   min loss = {optimum.loss:.3f}")

    if law is None:
        click.echo("\nAt least two budgets with three runs each are needed for a law.")
        return

    click.echo(f"\nScaling law: N_opt = {law.coefficient:.3g} · C^{law.exponent:.3f}")
    if size == "compute_params":
        click.echo(f"             D_opt = C / 6 N_opt ∝ C^{1 - law.exponent:.3f}")


def format_log_axis(axis: Axis) -> None:
    """Label a log axis at 1, 2 and 5 times powers of ten (e.g. 200k, 500k, 1M), so
    the labels of a narrow range neither overlap nor go missing."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(EngFormatter(sep=""))
    axis.set_minor_formatter(NullFormatter())


def chinchilla_optimum(budget: float) -> float:
    """The size Chinchilla's ~20 tokens per parameter predicts: C = 6 N (20 N)."""
    return math.sqrt(budget / 120)


def fit_parabola(sizes: np.ndarray, losses: np.ndarray) -> Polynomial | None:
    """Loss as a quadratic in log size; None with fewer than three runs."""
    if len(sizes) < 3:
        return None

    return Polynomial.fit(np.log(sizes), losses, deg=2)


def vertex(parabola: Polynomial) -> float:
    """The log size where the parabola has its minimum."""
    (root,) = parabola.deriv().roots()

    return float(root)


def save(figure: Figure, filename: str) -> None:
    path = PLOTS / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    click.echo(f"Saved {path}")


if __name__ == "__main__":
    main()
