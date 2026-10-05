from pathlib import Path

import click
from huggingface_hub import hf_hub_download

CURRENT_FOLDER = Path(__file__).parent
DATA = CURRENT_FOLDER.parent / "data" / "tinystories"
FILES = ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]


@click.command()
def main():
    """Download the TinyStories train and validation sets into the /data folder."""
    click.echo(f"Downloading data to {DATA}...")

    for filename in FILES:
        path = hf_hub_download(
            "roneneldan/TinyStories",
            filename,
            repo_type="dataset",
            local_dir=DATA,
        )

        click.echo(f"Downloaded {path}")


if __name__ == "__main__":
    main()
