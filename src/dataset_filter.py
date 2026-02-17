"""Dataset filtering logic for splitting datasets by year."""

import json
import re
from pathlib import Path
from typing import Any

import datasets


class FilterError(Exception):
    """Raised when dataset filtering fails."""

    pass


def extract_year_from_link(link: str | None) -> int | None:
    """Extract year from AOPS wiki link.

    Args:
        link: URL string, typically an AOPS wiki link.

    Returns:
        Extracted year as integer, or None if no valid year found.
    """
    if not link:
        return None

    # Pattern: 4-digit year followed by underscore and uppercase letter
    # e.g., "2004_AIME", "1983_AMC", "2015_USAMO"
    match = re.search(r"(\d{4})_[A-Z]", link)
    if match:
        return int(match.group(1))
    return None


def categorize_by_year(
    year: int | None, train_max: int, valid_year: int, test_min: int
) -> str:
    """Categorize a year into train/valid/test/unknown split.

    Args:
        year: The extracted year, or None if unknown.
        train_max: Maximum year for training split (inclusive).
        valid_year: Year for validation split.
        test_min: Minimum year for test split (inclusive).

    Returns:
        Split name: "train", "valid", "test", or "unknown".
    """
    if year is None:
        return "unknown"
    if year <= train_max:
        return "train"
    if year == valid_year:
        return "valid"
    if year >= test_min:
        return "test"
    return "unknown"


def filter_dataset(
    dataset: datasets.Dataset, year_boundaries: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Filter dataset rows into train/valid/test/unknown splits by year.

    Args:
        dataset: HuggingFace Dataset to filter.
        year_boundaries: Dictionary with keys:
            - train_max_year: Maximum year for training split
            - valid_year: Year for validation split
            - test_min_year: Minimum year for test split

    Returns:
        Dictionary mapping split names to lists of rows.
    """
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
        "unknown": [],
    }

    train_max = year_boundaries["train_max_year"]
    valid_year = year_boundaries["valid_year"]
    test_min = year_boundaries["test_min_year"]

    for row in dataset:
        row_dict = dict(row)
        link = row_dict.get("link", "")
        year = extract_year_from_link(link)

        # Add metadata
        row_dict["_extracted_year"] = year

        # Categorize and add to appropriate split
        split_name = categorize_by_year(year, train_max, valid_year, test_min)
        splits[split_name].append(row_dict)

    return splits


def write_splits(
    splits: dict[str, list[dict[str, Any]]], output_dir: Path | str
) -> dict[str, Path]:
    """Write split data to JSONL files.

    Args:
        splits: Dictionary mapping split names to lists of rows.
        output_dir: Directory to write JSONL files to.

    Returns:
        Dictionary mapping split names to output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    for split_name, rows in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        output_paths[split_name] = output_path

    return output_paths


def discover_splits(dataset_dir: Path) -> list[str]:
    """Discover available splits in a dataset directory.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        List of split names found in the directory.

    Raises:
        FilterError: If dataset directory doesn't exist or contains no splits.
    """
    if not dataset_dir.exists():
        raise FilterError(f"Dataset directory not found: {dataset_dir}")
    splits = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    if not splits:
        raise FilterError(f"No splits found in dataset directory: {dataset_dir}")
    return splits


def load_split(split_path: Path) -> datasets.Dataset:
    """Load dataset split from disk.

    Args:
        split_path: Path to the split directory.

    Returns:
        Loaded Dataset object.

    Raises:
        FilterError: If split directory doesn't exist.
    """
    if not split_path.exists():
        raise FilterError(f"Split directory not found: {split_path}")

    result = datasets.load_from_disk(str(split_path))
    assert isinstance(result, datasets.Dataset)
    return result


def filter_and_split_dataset(config: dict[str, Any]) -> dict[str, Path]:
    """Filter dataset by year and write to JSONL files.

    Auto-discovers all splits in the dataset directory and processes them,
    merging results into year-based output splits.

    Args:
        config: Configuration dictionary with keys:
            - input_data_dir: Base directory for input data
            - output_data_dir: Base directory for output data
            - dataset_name: Name of the dataset (e.g., "di-zhang-fdu/AOPS")
            - year_boundaries: Dictionary with train_max_year, valid_year, test_min_year

    Returns:
        Dictionary mapping split names to output file paths.

    Raises:
        FilterError: If input dataset doesn't exist or has no splits.
    """
    input_data_dir = Path(config["input_data_dir"]).resolve()
    output_data_dir = Path(config["output_data_dir"]).resolve()
    dataset_name = config["dataset_name"]
    year_boundaries = config["year_boundaries"]

    # Build dataset directory path and discover splits
    dataset_dir = input_data_dir / dataset_name
    input_splits = discover_splits(dataset_dir)

    # Initialize combined output splits
    combined_splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
        "unknown": [],
    }

    # Process each input split
    for input_split in input_splits:
        input_path = dataset_dir / input_split
        dataset = load_split(input_path)
        splits = filter_dataset(dataset, year_boundaries)

        # Merge into combined splits
        for split_name, rows in splits.items():
            combined_splits[split_name].extend(rows)

    # Build output directory
    output_dir = output_data_dir / dataset_name

    # Write splits
    output_paths = write_splits(combined_splits, output_dir)

    return output_paths
