"""Train a BPE tokenizer on TinyStories and save its merges."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import click

from src.bpe import BPE

DATA = Path(__file__).parent.parent / "data" / "tinystories"
CORPUS = DATA / "TinyStoriesV2-GPT4-train.txt"
MERGES = DATA / "merges.json"
END_OF_TEXT = "<|endoftext|>"


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


def read_documents(path: Path) -> Iterator[str]:
    """Stream the stories one by one, splitting on the end-of-text marker."""
    with path.open(encoding="utf-8") as file:
        story: list[str] = []

        for line in file:
            if line.strip() == END_OF_TEXT:
                yield "".join(story).strip()
                story = []
            else:
                story.append(line)


if __name__ == "__main__":
    main()
