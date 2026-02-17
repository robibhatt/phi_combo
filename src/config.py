"""YAML configuration loader utility."""

from pathlib import Path
from typing import Any, Callable

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


def load_config(
    config_path: Path,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        validator: Optional validation function. If None, uses default
            validation for download configs. Pass a custom function
            for different config formats.

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

    if validator is None:
        _validate_download_config(config)
    else:
        validator(config)

    return config


def _validate_download_config(config: dict[str, Any]) -> None:
    """Validate download dataset configuration.

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


def validate_sampler_config(config: dict[str, Any]) -> None:
    """Validate sampler configuration.

    Args:
        config: The parsed configuration dictionary.

    Raises:
        ConfigError: If required fields are missing or invalid.
    """
    if "data_dir" not in config:
        raise ConfigError("Missing required field: 'data_dir'")

    if "dataset_name" not in config:
        raise ConfigError("Missing required field: 'dataset_name'")

    if "split" not in config:
        raise ConfigError("Missing required field: 'split'")

    if "sample_size" not in config:
        raise ConfigError("Missing required field: 'sample_size'")

    if not isinstance(config["sample_size"], int) or config["sample_size"] <= 0:
        raise ConfigError("'sample_size' must be a positive integer")
