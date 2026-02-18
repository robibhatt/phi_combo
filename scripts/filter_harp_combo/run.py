#!/usr/bin/env python
"""Entry point for filtering HARP dataset to counting & probability problems.

Usage:
    python scripts/filter_harp_combo/run.py

Configuration is auto-loaded from config.yaml in the same directory.

Output structure:
    data/processed/HARP_combo/
        train.jsonl    # counting_and_probability, year <= 2023
        valid.jsonl    # counting_and_probability, year >= 2024
"""

import json
import re
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_harp_combo_config
from src.dataset_filter import write_splits

# Script directory for auto-loading config
SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """Get path to config.yaml in the script directory."""
    return SCRIPT_DIR / "config.yaml"


def main() -> None:
    """Main entry point for HARP combo filtering script."""
    config_path = get_config_path()
    config = load_config(config_path, validator=validate_harp_combo_config)

    input_file = Path(config["input_file"]).resolve()
    output_data_dir = Path(config["output_data_dir"]).resolve()
    dataset_name = config["dataset_name"]
    subject_filter = config["subject_filter"]
    train_max_year = config["train_max_year"]
    valid_min_year = config["valid_min_year"]

    # Read and filter HARP.jsonl
    splits: dict[str, list[dict]] = {"train": [], "valid": []}

    with open(input_file, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("subject") != subject_filter:
                continue
            year_raw = row.get("year")
            if year_raw is not None:
                match = re.match(r"(\d{4})", str(year_raw))
                year = int(match.group(1)) if match else None
            else:
                year = None
            if year is not None and year <= train_max_year:
                splits["train"].append(row)
            elif year is not None and year >= valid_min_year:
                splits["valid"].append(row)

    # Write output
    output_dir = output_data_dir / dataset_name
    output_paths = write_splits(splits, output_dir)

    print("HARP combo dataset filtered successfully!")
    print(f"Subject filter: {subject_filter}")
    print(f"Output files:")
    for split_name, path in output_paths.items():
        count = len(splits[split_name])
        print(f"  {split_name}: {path} ({count} rows)")


if __name__ == "__main__":
    main()
