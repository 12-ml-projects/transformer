"""Generate text with the best model from the study and check it against the grammar."""

from pathlib import Path
from random import Random

import click
import nltk
import optuna
import torch
from nltk.grammar import PCFG

from experiments.simple_language import main as experiment
from src.data import language
from src.model.transformer import Decoder

LANGUAGE_SPEC = Path("tests/language_specs/language_1.txt")

Sentence = list[str]


@click.command()
@click.option("--n-tokens", default=3000, help="Number of tokens to generate.")
@click.option("--seed", default=0, help="Seed for the prompt and the sampling.")
def generate_text(n_tokens: int, seed: int) -> None:
    """Generates text and checks it against the grammar."""
    torch.manual_seed(seed)
    pcfg = language.load(LANGUAGE_SPEC)
    vocab = language.vocabulary(pcfg)

    model = load_best_model(pcfg)
    prompt, _ = language.sample(pcfg, Random(seed))

    prompt_ids = [vocab.index(token) for token in prompt]
    ids = generate(model, prompt_ids, n_tokens)[len(prompt_ids) :]
    sentences = split_sentences([vocab[i] for i in ids])

    report(prompt, sentences, is_grammatical(pcfg, sentences))


def load_best_model(pcfg: PCFG) -> Decoder:
    """Rebuild the best trial's model and load its saved weights."""
    study = optuna.load_study(
        study_name=experiment.EXPERIMENT_NAME,
        storage=f"sqlite:///{experiment.RESULTS_DIR / 'optuna.db'}",
    )
    model = experiment.make_model(study.best_params, pcfg)
    model.load_state_dict(
        torch.load(
            experiment.BEST_MODEL_PATH,
            map_location=experiment.device,
            weights_only=True,
        )
    )

    return model


@torch.no_grad()
def generate(model: torch.nn.Module, prompt: list[int], n_tokens: int) -> list[int]:
    """Sample n_tokens after the prompt, feeding the model at most its context."""
    model.eval()
    ids = torch.tensor([prompt], device=experiment.device)

    for _ in range(n_tokens):
        logits = model(ids[:, -experiment.CONTEXT_LENGTH :])[:, -1, :]
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return ids[0].tolist()


def split_sentences(tokens: list[str]) -> list[Sentence]:
    """Split on ".", dropping the unfinished sentence at the end."""
    sentences: list[Sentence] = []
    current: Sentence = []

    for token in tokens:
        current.append(token)
        if token == ".":
            sentences.append(current)
            current = []

    return sentences


def is_grammatical(pcfg: PCFG, sentences: list[Sentence]) -> list[bool]:
    parser = nltk.ChartParser(pcfg)

    return [next(parser.parse(sentence), None) is not None for sentence in sentences]


def report(prompt: Sentence, sentences: list[Sentence], grammatical: list[bool]):
    n_grammatical = sum(grammatical)

    click.echo(f"Prompt: {' '.join(prompt)}")
    click.echo(
        f"{n_grammatical}/{len(sentences)} sentences are grammatical "
        f"({n_grammatical / len(sentences):.1%})\n"
    )
    for sentence, ok in list(zip(sentences, grammatical))[:20]:
        click.echo(f"{'  ok ' if ok else '  BAD'}  {' '.join(sentence)}")


if __name__ == "__main__":
    generate_text()
