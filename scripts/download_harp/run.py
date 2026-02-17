#!/usr/bin/env python
"""Entry point for downloading the HARP dataset from GitHub.

Usage:
    python scripts/download_harp/run.py

Configuration is auto-loaded from config.yaml in the same directory.
"""

from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.github_downloader import download_harp_dataset, validate_harp_config

# Script directory for auto-loading config
SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """Get path to config.yaml in the script directory."""
    return SCRIPT_DIR / "config.yaml"


def main() -> None:
    """Main entry point for HARP dataset download script."""
    config_path = get_config_path()
    config = load_config(config_path, validator=validate_harp_config)
    download_harp_dataset(config)


if __name__ == "__main__":
    main()
