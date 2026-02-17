"""Dataset sampling logic for creating JSONL samples from HuggingFace datasets."""

import json
import random
from pathlib import Path
from typing import Any

import datasets


class SamplerError(Exception):
    """Raised when sampling fails."""

    pass


def get_split_path(data_dir: Path | str, dataset_name: str, split: str) -> Path:
    """Return path to split directory.

    Args:
        data_dir: Base data directory.
        dataset_name: Name of the dataset (e.g., "nvidia/OpenMathReasoning").
        split: Name of the split (e.g., "cot", "train").

    Returns:
        Path to the split directory.
    """
    return Path(data_dir) / dataset_name / split


def validate_split_exists(split_path: Path) -> None:
    """Validate that split directory exists.

    Args:
        split_path: Path to the split directory.

    Raises:
        SamplerError: If split directory doesn't exist.
    """
    if not split_path.exists():
        raise SamplerError(f"Split directory not found: {split_path}")


def load_split(split_path: Path) -> datasets.Dataset:
    """Load dataset split from disk.

    Args:
        split_path: Path to the split directory.

    Returns:
        Loaded Dataset object.
    """
    result = datasets.load_from_disk(str(split_path))
    assert isinstance(result, datasets.Dataset)
    return result


def sample_rows(
    dataset: datasets.Dataset, sample_size: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """Randomly sample rows from dataset.

    Args:
        dataset: Dataset to sample from.
        sample_size: Number of rows to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of dictionaries containing sampled rows.
    """
    dataset_size = len(dataset)
    actual_size = min(sample_size, dataset_size)

    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(dataset_size), actual_size)
    selected = dataset.select(indices)

    return [dict(row) for row in selected]


def write_jsonl(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write rows to JSONL file.

    Args:
        rows: List of dictionaries to write.
        output_path: Path to output file.
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sample_dataset(config: dict[str, Any]) -> Path:
    """Sample rows from a dataset split and save to JSONL.

    Args:
        config: Configuration dictionary with keys:
            - data_dir: Base data directory
            - dataset_name: Name of the dataset
            - split: Name of the split to sample
            - sample_size: Number of rows to sample

    Returns:
        Path to the created JSONL file.

    Raises:
        SamplerError: If dataset split doesn't exist.
    """
    data_dir = Path(config["data_dir"]).resolve()
    dataset_name = config["dataset_name"]
    split = config["split"]
    sample_size = config["sample_size"]

    split_path = get_split_path(data_dir, dataset_name, split)
    validate_split_exists(split_path)

    dataset = load_split(split_path)
    rows = sample_rows(dataset, sample_size)

    output_path = split_path / "sample.jsonl"
    write_jsonl(rows, output_path)

    return output_path
