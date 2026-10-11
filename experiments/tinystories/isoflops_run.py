"""Train one model of an IsoFLOPs sweep on TinyStories.

A run is defined by a compute budget C and a depth. The width grows with the depth
at a fixed aspect ratio, so every model has a sensible shape, and the training
recipe is fixed. The number of training tokens follows from C = 6 N D.
"""

import math
import time
from pathlib import Path
from typing import Any

import click
import mlflow
import torch
from torch.utils.data import DataLoader, Subset

from src.bpe import BPE
from src.data.tiny_stories import TinyStoriesDataset
from src.experiment import train
from src.experiment.optimizers import gpt3_adamw
from src.experiment.schedulers import cosine_schedule
from src.experiment.utils import get_device
from src.model.transformer import Decoder

EXPERIMENT_NAME = "tinystories-isoflops"
DATA = Path(__file__).parents[2] / "data" / "tinystories"

VOCAB_SIZE = 4097  # 4096 BPE tokens plus the end-of-text token
ASPECT_RATIO = 32  # d_model per layer: models grow in width and depth together
HEAD_DIM = 32
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WARMUP_FRACTION = 0.02
N_EVALUATIONS = 10
VALID_WINDOWS = 1_024


@click.command()
@click.option("--budget", type=float, required=True, help="Compute budget in FLOPs.")
@click.option("--n-layers", type=int, required=True, help="Depth of the model.")
@click.option("--sweep", default="v1", help="Tag grouping the runs of one sweep.")
@click.option(
    "--mlflow-tracking-uri",
    default="http://localhost:8888",
    help="MLflow tracking URI.",
)
@click.option(
    "--save-model",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save the trained model and its configuration to this .pt file.",
)
def main(
    budget: float,
    n_layers: int,
    sweep: str,
    mlflow_tracking_uri: str,
    save_model: Path | None,
) -> None:
    """Run one IsoFLOPs experiment on TinyStories for a budget and model depth."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"C={budget:.0e}-L={n_layers}", tags={"sweep": sweep}
    ):
        result = run(budget, n_layers, save_model)

    click.echo(result)


def run(
    budget: float, n_layers: int, save_model: Path | None = None
) -> dict[str, float]:
    """Train one model for the budget and log the result to the active MLflow run."""
    torch.manual_seed(0)
    device = get_device()

    tokenizer = BPE.load(DATA / "merges.json")
    train_data = load_split(tokenizer, "train")
    valid_data = load_split(tokenizer, "valid")
    if train_data.vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Expected a vocabulary of {VOCAB_SIZE}, got {train_data.vocab_size}"
        )

    config = model_config(n_layers)
    model = Decoder(**config).to(device)

    n_params = count_non_embedding_params(model)
    compute_params = count_compute_params(model)
    n_tokens = tokens_for_budget(budget, compute_params)
    total_steps = math.ceil(n_tokens / (BATCH_SIZE * CONTEXT_LENGTH))

    optimizer = gpt3_adamw(model, LEARNING_RATE)
    scheduler = cosine_schedule(
        optimizer, max(1, int(WARMUP_FRACTION * total_steps)), total_steps
    )

    mlflow.log_params(
        {
            "budget": budget,
            "n_layers": n_layers,
            "d_model": config["input_dim"],
            "n_params": n_params,
            "compute_params": compute_params,
            "n_tokens": n_tokens,
            "context_length": CONTEXT_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
    )

    eval_every = max(1, n_tokens // N_EVALUATIONS)
    start = time.perf_counter()
    valid_loss = math.inf

    for idx, (train_loss, valid_loss) in enumerate(
        train(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(),
            train_loader=DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
            valid_loader=DataLoader(
                Subset(valid_data, range(VALID_WINDOWS)), batch_size=BATCH_SIZE
            ),
            n_tokens=n_tokens,
            eval_every=eval_every,
            device=device,
            scheduler=scheduler,
            max_grad_norm=1.0,
        )
    ):
        mlflow.log_metrics(
            {"train_loss": train_loss, "valid_loss": valid_loss},
            step=min((idx + 1) * eval_every, n_tokens),
        )

    time_delta = time.perf_counter() - start
    result = {
        "valid_loss": valid_loss,
        "n_params": n_params,
        "n_tokens": n_tokens,
        "flops": 6 * compute_params * n_tokens,
        "tokens_per_second": n_tokens / time_delta,
        "time_delta": time_delta,
    }
    mlflow.log_metrics(result)

    if save_model is not None:
        # The configuration travels with the weights, so the file alone is enough to
        # rebuild the model: Decoder(**checkpoint["config"]).
        save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "state_dict": model.state_dict()}, save_model)

    return result


def model_config(n_layers: int) -> dict[str, Any]:
    """The GPT-2-style decoder of a given depth, with its width set by ASPECT_RATIO
    and one head per HEAD_DIM dimensions."""
    d_model = ASPECT_RATIO * n_layers

    return {
        "n_layers": n_layers,
        "max_length": CONTEXT_LENGTH,
        "input_dim": d_model,
        "vocab_size": VOCAB_SIZE,
        "heads": d_model // HEAD_DIM,
        "dropout": 0.0,  # we only see one pass through the data
        "gpt_2_like": True,
    }


def load_split(tokenizer: BPE, split: str) -> TinyStoriesDataset:
    dataset = TinyStoriesDataset(
        tokenizer,
        corpus=DATA / f"TinyStoriesV2-GPT4-{split}.txt",
        token_file=DATA / f"{split}.bin",
        context_length=CONTEXT_LENGTH,
    )
    dataset.load()

    return dataset


def count_non_embedding_params(model: Decoder) -> int:
    """N as in Kaplan et al.: everything except the token embedding and the output
    layer, whose size depends on the vocabulary rather than on the model."""
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith(("_input_embedding.", "_linear_final_layer."))
    )


def count_compute_params(model: Decoder) -> int:
    """The parameters that cost compute on every token: N plus the output layer.

    N leaves the output layer out as a convention for model size, but it is still a
    matrix multiplication per token. For small models it costs more than the blocks
    (vocab_size x d_model against 12 d_model^2 per block), so leaving it out of the
    compute would hand the smallest models, which see the most tokens, extra compute.
    The token embedding is a lookup and costs no multiplications.
    """
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("_input_embedding.")
    )


def tokens_for_budget(budget: float, compute_params: int) -> int:
    """D from C = 6 N D, with N the parameters that cost compute per token: about 6
    FLOPs per such parameter per token, forward and backward."""
    return int(budget / (6 * compute_params))


if __name__ == "__main__":
    main()
