from pathlib import Path

import click
from huggingface_hub import hf_hub_download

CURRENT_FOLDER = Path(__file__).parent
DATA = CURRENT_FOLDER.parent / "data" / "tinystories"


@click.command()
def main():
    """Download TinyStories into the /data folder."""
    click.echo(f"Downloading data to {DATA}...")

    path = hf_hub_download(
        "roneneldan/TinyStories",
        "TinyStoriesV2-GPT4-train.txt",
        repo_type="dataset",
        local_dir=DATA,
    )

    click.echo(f"Downloaded data to {path}")

    return


if __name__ == "__main__":
    main()
