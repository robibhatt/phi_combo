"""Tests for scripts/download_dataset/run.py - Entry point script."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import yaml


class TestScriptRun:
    """Tests for the download_dataset script."""

    def test_script_loads_config_from_same_directory(self, tmp_path, monkeypatch):
        """Verify YAML auto-loaded from script directory."""
        # Create a mock script directory structure
        script_dir = tmp_path / "scripts" / "download_dataset"
        script_dir.mkdir(parents=True)

        # Create config file
        config_data = {
            "data_dir": str(tmp_path / "data"),
            "datasets": [{"name": "test_dataset"}]
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Import and test the get_config_path function
        from scripts.download_dataset.run import get_config_path

        # Mock __file__ to point to our test script directory
        with patch("scripts.download_dataset.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == config_file

    def test_script_runs_end_to_end(self, tmp_path, monkeypatch):
        """Integration test for full workflow."""
        # Setup: Create config file
        script_dir = tmp_path / "scripts" / "download_dataset"
        script_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        config_data = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "imdb", "subset": None}]
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Mock the download to avoid actual network calls
        mock_dataset = MagicMock()
        with patch("scripts.download_dataset.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_loader.datasets.load_dataset", return_value=mock_dataset):
            from scripts.download_dataset.run import main
            main()

        # Verify download was triggered
        mock_dataset.save_to_disk.assert_called_once()

    def test_script_handles_missing_config(self, tmp_path, monkeypatch):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "download_dataset"
        script_dir.mkdir(parents=True)

        # No config file created

        from scripts.download_dataset.run import main
        from src.config import ConfigError

        with patch("scripts.download_dataset.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_skips_existing_datasets(self, tmp_path, monkeypatch):
        """Skip download when dataset already exists."""
        # Setup directories
        script_dir = tmp_path / "scripts" / "download_dataset"
        script_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create existing dataset
        dataset_dir = data_dir / "imdb"
        dataset_dir.mkdir()
        (dataset_dir / "dataset_info.json").touch()

        config_data = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "imdb", "subset": None}]
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("scripts.download_dataset.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_loader.datasets.load_dataset") as mock_load:
            from scripts.download_dataset.run import main
            main()

            # Should not download since dataset exists
            mock_load.assert_not_called()
