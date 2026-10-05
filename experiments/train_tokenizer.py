"""Train a BPE tokenizer on TinyStories and save its merges."""

from itertools import islice
from pathlib import Path

import click

from src.bpe import BPE
from src.data.tiny_stories import read_documents

DATA = Path(__file__).parent.parent / "data" / "tinystories"
CORPUS = DATA / "TinyStoriesV2-GPT4-train.txt"
MERGES = DATA / "merges.json"


@click.command()
@click.option("--vocab-size", default=4096, help="Vocabulary size, including bytes.")
@click.option("--max-documents", type=int, help="Only use the first N stories.")
def main(vocab_size: int, max_documents: int | None):
    """Train a tokenizer on TinyStories and save it to data/tinystories."""
    documents = islice(read_documents(CORPUS), max_documents)

    with click.progressbar(
        documents, length=max_documents, label="Reading stories"
    ) as bar:
        tokenizer = BPE.train(bar, n_merges=vocab_size - 256)

    tokenizer.save(MERGES)

    click.echo(f"Saved {tokenizer.vocab_size} tokens to {MERGES}")


if __name__ == "__main__":
    main()
