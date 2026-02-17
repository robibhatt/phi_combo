#!/usr/bin/env python
"""Entry point for sampling HuggingFace dataset splits.

Usage:
    python scripts/sample_dataset/run.py

Configuration is auto-loaded from config.yaml in the same directory.
"""

from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_sampler_config
from src.dataset_sampler import sample_dataset

# Script directory for auto-loading config
SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """Get path to config.yaml in the script directory."""
    return SCRIPT_DIR / "config.yaml"


def main() -> None:
    """Main entry point for dataset sampling script."""
    config_path = get_config_path()
    config = load_config(config_path, validator=validate_sampler_config)
    output_path = sample_dataset(config)
    print(f"Sample saved to: {output_path}")


if __name__ == "__main__":
    main()
