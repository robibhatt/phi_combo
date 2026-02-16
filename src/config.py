"""YAML configuration loader utility."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


def load_config(config_path: Path) -> dict[str, Any]:
    """Load dataset configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        ConfigError: If the file is not found or validation fails.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)

    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the configuration dictionary.

    Args:
        config: The parsed configuration dictionary.

    Raises:
        ConfigError: If required fields are missing or invalid.
    """
    if "data_dir" not in config:
        raise ConfigError("Missing required field: 'data_dir'")

    if "datasets" not in config:
        raise ConfigError("Missing required field: 'datasets'")

    if not isinstance(config["datasets"], list):
        raise ConfigError("'datasets' must be a list")

    for i, dataset in enumerate(config["datasets"]):
        if "name" not in dataset:
            raise ConfigError(f"Dataset at index {i} missing required field: 'name'")
