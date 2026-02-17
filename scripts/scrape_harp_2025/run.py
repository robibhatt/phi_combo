#!/usr/bin/env python
"""Entry point for scraping 2025 AMC/AIME problems from AoPS wiki.

Usage:
    python scripts/scrape_harp_2025/run.py

Configuration is auto-loaded from config.yaml in the same directory.
"""

from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_scraper_config
from src.harp_scraper import scrape_all_2025_problems, save_to_jsonl

# Script directory for auto-loading config
SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """Get path to config.yaml in the script directory."""
    return SCRIPT_DIR / "config.yaml"


def main() -> None:
    """Main entry point for HARP 2025 scraper script."""
    config_path = get_config_path()
    config = load_config(config_path, validator=validate_scraper_config)

    print("Starting HARP 2025 scraper...", flush=True)
    print(f"Output: {config['output_dir']}/{config['output_filename']}", flush=True)

    problems, stats = scrape_all_2025_problems(config)

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir

    output_path = output_dir / config["output_filename"]
    save_to_jsonl(problems, output_path)

    print(f"\nDone! Saved {len(problems)} problems to {output_path}", flush=True)
    print(
        f"Session: {stats['found']} new, {stats['skipped']} resumed, "
        f"{stats['not_found']} not found, {stats['errors']} errors",
        flush=True,
    )


if __name__ == "__main__":
    main()
