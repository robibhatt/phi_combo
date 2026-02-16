"""Tests for src/config.py - YAML configuration loader."""

import pytest
from pathlib import Path
import tempfile
import yaml

from src.config import load_config, ConfigError


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_yaml(self, tmp_path):
        """Load and parse YAML config file."""
        config_data = {
            "data_dir": "../../data/hugging_face",
            "datasets": [
                {"name": "imdb", "subset": None}
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)

        assert result["data_dir"] == "../../data/hugging_face"
        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["name"] == "imdb"

    def test_config_missing_file_raises_error(self, tmp_path):
        """Handle missing config file gracefully."""
        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigError) as exc_info:
            load_config(missing_file)

        assert "not found" in str(exc_info.value).lower()

    def test_config_validates_required_fields(self, tmp_path):
        """Validate required fields present."""
        # Missing 'datasets' field
        config_data = {"data_dir": "../../data/hugging_face"}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)

        assert "datasets" in str(exc_info.value).lower()

    def test_config_validates_data_dir_required(self, tmp_path):
        """Validate data_dir field is required."""
        config_data = {"datasets": [{"name": "imdb"}]}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)

        assert "data_dir" in str(exc_info.value).lower()

    def test_config_validates_dataset_name_required(self, tmp_path):
        """Validate each dataset has a name."""
        config_data = {
            "data_dir": "../../data/hugging_face",
            "datasets": [{"subset": "test"}]  # missing 'name'
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)

        assert "name" in str(exc_info.value).lower()

    def test_config_handles_optional_fields(self, tmp_path):
        """Handle optional subset field."""
        config_data = {
            "data_dir": "../../data/hugging_face",
            "datasets": [
                {"name": "squad", "subset": "plain_text"},
                {"name": "imdb"}  # no subset
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)

        assert result["datasets"][0]["subset"] == "plain_text"
        assert result["datasets"][1].get("subset") is None
