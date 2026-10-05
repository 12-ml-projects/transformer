"""Encode the TinyStories train and validation sets into uint16 token files."""

from pathlib import Path

import click

from src.bpe import BPE
from src.data.tiny_stories import TinyStoriesDataset

DATA = Path(__file__).parent.parent / "data" / "tinystories"
MERGES = DATA / "merges.json"
SPLITS = {
    "train": DATA / "TinyStoriesV2-GPT4-train.txt",
    "valid": DATA / "TinyStoriesV2-GPT4-valid.txt",
}


@click.command()
@click.option("--overwrite", is_flag=True, help="Re-encode existing token files.")
def main(overwrite: bool):
    """Encode each split once into data/tinystories/<split>.bin."""
    tokenizer = BPE.load(MERGES)

    for split, corpus in SPLITS.items():
        token_file = DATA / f"{split}.bin"
        click.echo(f"Encoding {corpus.name} into {token_file.name}...")

        dataset = TinyStoriesDataset(tokenizer, corpus, token_file, context_length=1)
        dataset.load(skip_existing=not overwrite)

        click.echo(f"{split}: {dataset.n_tokens:,} tokens")


if __name__ == "__main__":
    main()
