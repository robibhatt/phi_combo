"""Dataset download logic for Hugging Face datasets."""

from pathlib import Path
from typing import Any

import datasets


def get_data_dir(config: dict[str, Any]) -> Path:
    """Return path to data directory, creating if needed.

    Args:
        config: Configuration dictionary with 'data_dir' key.

    Returns:
        Resolved Path to the data directory.
    """
    data_dir = Path(config["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def dataset_exists(data_dir: Path, dataset_name: str) -> bool:
    """Check if dataset already exists locally.

    A dataset is considered to exist if its directory contains at least one file.

    Args:
        data_dir: Path to the data directory.
        dataset_name: Name of the dataset to check.

    Returns:
        True if dataset exists with content, False otherwise.
    """
    dataset_path = data_dir / dataset_name

    if not dataset_path.exists():
        return False

    # Check if directory has any files
    return any(dataset_path.iterdir())


def download_dataset(config: dict[str, Any]) -> None:
    """Download all datasets from config if not present.

    Args:
        config: Configuration dictionary with 'data_dir' and 'datasets' keys.
    """
    data_dir = get_data_dir(config)

    for dataset_config in config["datasets"]:
        name = dataset_config["name"]
        subset = dataset_config.get("subset")

        if dataset_exists(data_dir, name):
            print(f"Dataset '{name}' already exists, skipping download.")
            continue

        print(f"Downloading dataset '{name}'...")
        dataset = datasets.load_dataset(name, subset)

        save_path = data_dir / name
        dataset.save_to_disk(str(save_path))
        print(f"Dataset '{name}' saved to {save_path}")
