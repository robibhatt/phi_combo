#!/usr/bin/env python
"""Entry point for filtering di-zhang-fdu/AOPS dataset by year.

Usage:
    python scripts/filter_di_zhang/run.py

Configuration is auto-loaded from config.yaml in the same directory.

Output structure:
    data/processed/di-zhang-fdu/AOPS/
        train.jsonl    # year <= 2023
        valid.jsonl    # year == 2024
        test.jsonl     # year >= 2025
        unknown.jsonl  # year not extractable
"""

from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_filter_config
from src.dataset_filter import filter_and_split_dataset

# Script directory for auto-loading config
SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """Get path to config.yaml in the script directory."""
    return SCRIPT_DIR / "config.yaml"


def main() -> None:
    """Main entry point for dataset filtering script."""
    config_path = get_config_path()
    config = load_config(config_path, validator=validate_filter_config)
    output_paths = filter_and_split_dataset(config)

    print("Dataset filtered successfully!")
    print("Output files:")
    for split_name, path in output_paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
