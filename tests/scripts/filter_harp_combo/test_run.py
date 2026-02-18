"""Tests for scripts/filter_harp_combo/run.py - HARP combo filtering script."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
import yaml

from src.config import ConfigError


def _make_harp_row(year, subject, problem="p", answer="a"):
    """Create a minimal HARP JSONL row."""
    return {
        "year": year,
        "contest": "AMC",
        "number": 1,
        "subject": subject,
        "problem": problem,
        "answer": answer,
        "level": 3,
    }


def _write_jsonl(path, rows):
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestScriptRun:
    """Tests for the filter_harp_combo script."""

    def test_get_config_path_returns_correct_path(self, tmp_path):
        """Verify get_config_path returns path to config.yaml in script directory."""
        script_dir = tmp_path / "scripts" / "filter_harp_combo"
        script_dir.mkdir(parents=True)

        from scripts.filter_harp_combo.run import get_config_path

        with patch("scripts.filter_harp_combo.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == script_dir / "config.yaml"

    def test_main_filters_by_subject_and_splits_by_year(self, tmp_path, capsys):
        """Filter to counting_and_probability and split into train/valid."""
        script_dir = tmp_path / "scripts" / "filter_harp_combo"
        script_dir.mkdir(parents=True)

        # Create input JSONL with mixed subjects and years
        input_file = tmp_path / "data" / "HARP.jsonl"
        rows = [
            _make_harp_row(2020, "counting_and_probability", "train_cp"),
            _make_harp_row(2023, "counting_and_probability", "train_cp2"),
            _make_harp_row(2024, "counting_and_probability", "valid_cp"),
            _make_harp_row(2025, "counting_and_probability", "valid_cp2"),
            _make_harp_row(2020, "algebra", "train_alg"),
            _make_harp_row(2024, "geometry", "valid_geom"),
        ]
        _write_jsonl(input_file, rows)

        output_dir = tmp_path / "output"

        config_data = {
            "input_file": str(input_file),
            "output_data_dir": str(output_dir),
            "dataset_name": "HARP_combo",
            "subject_filter": "counting_and_probability",
            "train_max_year": 2023,
            "valid_min_year": 2024,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("scripts.filter_harp_combo.run.SCRIPT_DIR", script_dir):
            from scripts.filter_harp_combo.run import main
            main()

        expected_dir = output_dir / "HARP_combo"

        # Check train split
        train_lines = (expected_dir / "train.jsonl").read_text().strip().split("\n")
        train_rows = [json.loads(line) for line in train_lines]
        assert len(train_rows) == 2
        assert all(r["subject"] == "counting_and_probability" for r in train_rows)
        assert all(r["year"] <= 2023 for r in train_rows)

        # Check valid split
        valid_lines = (expected_dir / "valid.jsonl").read_text().strip().split("\n")
        valid_rows = [json.loads(line) for line in valid_lines]
        assert len(valid_rows) == 2
        assert all(r["subject"] == "counting_and_probability" for r in valid_rows)
        assert all(r["year"] >= 2024 for r in valid_rows)

        # Check printed output
        captured = capsys.readouterr()
        assert "HARP combo" in captured.out
        assert "2 rows" in captured.out

    def test_main_handles_missing_config(self, tmp_path):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "filter_harp_combo"
        script_dir.mkdir(parents=True)

        from scripts.filter_harp_combo.run import main

        with patch("scripts.filter_harp_combo.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_main_empty_after_filter(self, tmp_path):
        """Handle case where no rows match the subject filter."""
        script_dir = tmp_path / "scripts" / "filter_harp_combo"
        script_dir.mkdir(parents=True)

        input_file = tmp_path / "data" / "HARP.jsonl"
        rows = [
            _make_harp_row(2020, "algebra", "alg1"),
            _make_harp_row(2024, "geometry", "geom1"),
        ]
        _write_jsonl(input_file, rows)

        output_dir = tmp_path / "output"

        config_data = {
            "input_file": str(input_file),
            "output_data_dir": str(output_dir),
            "dataset_name": "HARP_combo",
            "subject_filter": "counting_and_probability",
            "train_max_year": 2023,
            "valid_min_year": 2024,
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("scripts.filter_harp_combo.run.SCRIPT_DIR", script_dir):
            from scripts.filter_harp_combo.run import main
            main()

        expected_dir = output_dir / "HARP_combo"
        assert (expected_dir / "train.jsonl").read_text() == ""
        assert (expected_dir / "valid.jsonl").read_text() == ""
