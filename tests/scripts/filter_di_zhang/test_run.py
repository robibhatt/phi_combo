"""Tests for scripts/filter_di_zhang/run.py - Entry point script."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.dataset_filter import FilterError


class TestScriptRun:
    """Tests for the filter_di_zhang script."""

    def test_get_config_path_returns_correct_path(self, tmp_path):
        """Verify get_config_path returns path to config.yaml in script directory."""
        script_dir = tmp_path / "scripts" / "filter_di_zhang"
        script_dir.mkdir(parents=True)

        from scripts.filter_di_zhang.run import get_config_path

        with patch("scripts.filter_di_zhang.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == script_dir / "config.yaml"

    def test_main_calls_filter_and_split_dataset(self, tmp_path):
        """Verify main() calls filter_and_split_dataset with loaded config."""
        script_dir = tmp_path / "scripts" / "filter_di_zhang"
        script_dir.mkdir(parents=True)

        dataset_dir = tmp_path / "input" / "di-zhang-fdu" / "AOPS"
        (dataset_dir / "train").mkdir(parents=True)

        output_dir = tmp_path / "output"

        config_data = {
            "input_data_dir": str(tmp_path / "input"),
            "output_data_dir": str(output_dir),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_rows = [
            {"link": "https://aops.com/2020_AMC_Problems/1", "problem": "p1"},
            {"link": "https://aops.com/2024_AMC_Problems/2", "problem": "p2"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=2)

        with patch("scripts.filter_di_zhang.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_filter.load_split", return_value=mock_dataset):
            from scripts.filter_di_zhang.run import main
            main()

        expected_output_dir = output_dir / "di-zhang-fdu" / "AOPS"
        assert (expected_output_dir / "train.jsonl").exists()
        assert (expected_output_dir / "valid.jsonl").exists()

    def test_script_handles_missing_config(self, tmp_path):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "filter_di_zhang"
        script_dir.mkdir(parents=True)

        from scripts.filter_di_zhang.run import main
        from src.config import ConfigError

        with patch("scripts.filter_di_zhang.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_handles_missing_input_dataset(self, tmp_path):
        """Handle missing input dataset gracefully."""
        script_dir = tmp_path / "scripts" / "filter_di_zhang"
        script_dir.mkdir(parents=True)

        config_data = {
            "input_data_dir": str(tmp_path / "nonexistent"),
            "output_data_dir": str(tmp_path / "output"),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        from scripts.filter_di_zhang.run import main

        with patch("scripts.filter_di_zhang.run.SCRIPT_DIR", script_dir):
            with pytest.raises(FilterError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_runs_end_to_end(self, tmp_path, capsys):
        """Integration test for full workflow."""
        script_dir = tmp_path / "scripts" / "filter_di_zhang"
        script_dir.mkdir(parents=True)

        dataset_dir = tmp_path / "input" / "di-zhang-fdu" / "AOPS"
        (dataset_dir / "train").mkdir(parents=True)

        output_dir = tmp_path / "output"

        config_data = {
            "input_data_dir": str(tmp_path / "input"),
            "output_data_dir": str(output_dir),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_rows = [
            {"link": "https://aops.com/2000_AMC_Problems/1", "problem": "old"},
            {"link": "https://aops.com/2024_IMO_Problems/2", "problem": "valid"},
            {"link": "https://aops.com/2025_AIME_Problems/3", "problem": "test"},
            {"link": "https://example.com/unknown", "problem": "unknown"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=4)

        with patch("scripts.filter_di_zhang.run.SCRIPT_DIR", script_dir), \
             patch("src.dataset_filter.load_split", return_value=mock_dataset):
            from scripts.filter_di_zhang.run import main
            main()

        captured = capsys.readouterr()
        assert "train.jsonl" in captured.out or "valid.jsonl" in captured.out

        expected_output_dir = output_dir / "di-zhang-fdu" / "AOPS"

        # Verify train split
        train_lines = (expected_output_dir / "train.jsonl").read_text().strip().split("\n")
        assert len(train_lines) == 1
        train_row = json.loads(train_lines[0])
        assert train_row["_extracted_year"] == 2000

        # Verify valid split
        valid_lines = (expected_output_dir / "valid.jsonl").read_text().strip().split("\n")
        assert len(valid_lines) == 1
        valid_row = json.loads(valid_lines[0])
        assert valid_row["_extracted_year"] == 2024

        # Verify test split
        test_lines = (expected_output_dir / "test.jsonl").read_text().strip().split("\n")
        assert len(test_lines) == 1
        test_row = json.loads(test_lines[0])
        assert test_row["_extracted_year"] == 2025

        # Verify unknown split
        unknown_lines = (expected_output_dir / "unknown.jsonl").read_text().strip().split("\n")
        assert len(unknown_lines) == 1
        unknown_row = json.loads(unknown_lines[0])
        assert unknown_row["_extracted_year"] is None
