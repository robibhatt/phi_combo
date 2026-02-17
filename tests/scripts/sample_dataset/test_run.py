"""Tests for scripts/sample_dataset/run.py - Entry point script."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.dataset_sampler import SamplerError


class TestScriptRun:
    """Tests for the sample_dataset script."""

    def test_script_loads_config_from_same_directory(self, tmp_path):
        """Verify YAML auto-loaded from script directory."""
        # Create a mock script directory structure
        script_dir = tmp_path / "scripts" / "sample_dataset"
        script_dir.mkdir(parents=True)

        # Create config file
        config_data = {
            "data_dir": str(tmp_path / "data"),
            "dataset_name": "test/dataset",
            "split": "train",
            "sample_size": 10,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Import and test the get_config_path function
        from scripts.sample_dataset.run import get_config_path

        # Mock SCRIPT_DIR to point to our test script directory
        with patch("scripts.sample_dataset.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == config_file

    def test_main_calls_sample_dataset(self, tmp_path):
        """Verify main() calls sample_dataset with loaded config."""
        # Setup script directory
        script_dir = tmp_path / "scripts" / "sample_dataset"
        script_dir.mkdir(parents=True)

        # Setup data directory with split
        data_dir = tmp_path / "data"
        split_dir = data_dir / "test/dataset" / "train"
        split_dir.mkdir(parents=True)

        config_data = {
            "data_dir": str(data_dir),
            "dataset_name": "test/dataset",
            "split": "train",
            "sample_size": 5,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.select = MagicMock(return_value=[
            {"id": i, "text": f"row_{i}"} for i in range(5)
        ])

        with patch("scripts.sample_dataset.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_sampler.load_split", return_value=mock_dataset):
            from scripts.sample_dataset.run import main
            main()

        # Verify JSONL was created
        output_path = split_dir / "sample.jsonl"
        assert output_path.exists()

        # Verify content is valid JSONL
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "id" in parsed

    def test_script_handles_missing_config(self, tmp_path):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "sample_dataset"
        script_dir.mkdir(parents=True)

        # No config file created

        from scripts.sample_dataset.run import main
        from src.config import ConfigError

        with patch("scripts.sample_dataset.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_handles_missing_split(self, tmp_path):
        """Handle missing split directory gracefully."""
        script_dir = tmp_path / "scripts" / "sample_dataset"
        script_dir.mkdir(parents=True)

        config_data = {
            "data_dir": str(tmp_path / "data"),
            "dataset_name": "missing/dataset",
            "split": "train",
            "sample_size": 10,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        from scripts.sample_dataset.run import main

        with patch("scripts.sample_dataset.run.SCRIPT_DIR", script_dir):
            with pytest.raises(SamplerError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_runs_end_to_end(self, tmp_path, capsys):
        """Integration test for full workflow."""
        # Setup script directory
        script_dir = tmp_path / "scripts" / "sample_dataset"
        script_dir.mkdir(parents=True)

        # Setup data directory with split
        data_dir = tmp_path / "data"
        split_dir = data_dir / "nvidia/OpenMathReasoning" / "cot"
        split_dir.mkdir(parents=True)

        config_data = {
            "data_dir": str(data_dir),
            "dataset_name": "nvidia/OpenMathReasoning",
            "split": "cot",
            "sample_size": 3,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.select = MagicMock(return_value=[
            {"problem": "2+2=?", "answer": "4"},
            {"problem": "3+3=?", "answer": "6"},
            {"problem": "5+5=?", "answer": "10"},
        ])

        with patch("scripts.sample_dataset.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_sampler.load_split", return_value=mock_dataset):
            from scripts.sample_dataset.run import main
            main()

        # Verify output path printed
        captured = capsys.readouterr()
        assert "sample.jsonl" in captured.out

        # Verify file created with correct content
        output_path = split_dir / "sample.jsonl"
        assert output_path.exists()

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3

        first_row = json.loads(lines[0])
        assert first_row["problem"] == "2+2=?"
        assert first_row["answer"] == "4"
