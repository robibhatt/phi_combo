"""Tests for src/dataset_sampler.py - Dataset sampling logic."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.dataset_sampler import (
    SamplerError,
    get_split_path,
    validate_split_exists,
    load_split,
    sample_rows,
    write_jsonl,
    sample_dataset,
)


class TestGetSplitPath:
    """Tests for get_split_path function."""

    def test_get_split_path_returns_correct_path(self, tmp_path):
        """Return correct path to split directory."""
        data_dir = tmp_path / "data"
        dataset_name = "nvidia/OpenMathReasoning"
        split = "cot"

        result = get_split_path(data_dir, dataset_name, split)

        expected = data_dir / "nvidia/OpenMathReasoning" / "cot"
        assert result == expected

    def test_get_split_path_handles_different_splits(self, tmp_path):
        """Handle various split names."""
        data_dir = tmp_path / "data"
        dataset_name = "test/dataset"

        for split in ["train", "test", "validation", "cot", "tir"]:
            result = get_split_path(data_dir, dataset_name, split)
            assert result == data_dir / "test/dataset" / split


class TestValidateSplitExists:
    """Tests for validate_split_exists function."""

    def test_validate_split_exists_raises_on_missing(self, tmp_path):
        """Raise SamplerError when split directory doesn't exist."""
        missing_path = tmp_path / "nonexistent" / "split"

        with pytest.raises(SamplerError) as exc_info:
            validate_split_exists(missing_path)

        assert "not found" in str(exc_info.value).lower()

    def test_validate_split_exists_passes_on_valid(self, tmp_path):
        """Pass silently when split directory exists."""
        split_path = tmp_path / "valid_split"
        split_path.mkdir(parents=True)

        # Should not raise
        validate_split_exists(split_path)


class TestSampleRows:
    """Tests for sample_rows function."""

    def test_sample_rows_returns_correct_count(self):
        """Return exactly sample_size rows."""
        # Create mock dataset with 100 rows
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.select = MagicMock(return_value=[
            {"text": f"row_{i}"} for i in range(10)
        ])

        result = sample_rows(mock_dataset, 10, seed=42)

        assert len(result) == 10
        mock_dataset.select.assert_called_once()

    def test_sample_rows_handles_oversized_request(self):
        """Return all rows when sample_size > dataset size."""
        # Create mock dataset with only 5 rows
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_dataset.select = MagicMock(return_value=[
            {"text": f"row_{i}"} for i in range(5)
        ])

        result = sample_rows(mock_dataset, 100, seed=42)

        assert len(result) == 5

    def test_sample_rows_returns_dicts(self):
        """Return list of dictionaries."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.select = MagicMock(return_value=[
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ])

        result = sample_rows(mock_dataset, 2, seed=42)

        assert all(isinstance(row, dict) for row in result)
        assert result[0] == {"id": 1, "text": "hello"}

    def test_sample_rows_with_seed_is_reproducible(self):
        """Same seed produces same sample."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        # Track calls to capture indices
        selected_indices = []
        def capture_select(indices):
            selected_indices.append(list(indices))
            return [{"id": i} for i in indices]

        mock_dataset.select = capture_select

        sample_rows(mock_dataset, 10, seed=42)
        sample_rows(mock_dataset, 10, seed=42)

        assert selected_indices[0] == selected_indices[1]


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_write_jsonl_creates_file(self, tmp_path):
        """Create JSONL file at specified path."""
        output_path = tmp_path / "output.jsonl"
        rows = [{"id": 1}, {"id": 2}]

        write_jsonl(rows, output_path)

        assert output_path.exists()

    def test_write_jsonl_overwrites_existing(self, tmp_path):
        """Overwrite existing file."""
        output_path = tmp_path / "output.jsonl"
        output_path.write_text("old content")
        rows = [{"id": 1}]

        write_jsonl(rows, output_path)

        content = output_path.read_text()
        assert "old content" not in content
        assert "id" in content

    def test_write_jsonl_format_is_valid(self, tmp_path):
        """Write valid JSONL format (one JSON object per line)."""
        output_path = tmp_path / "output.jsonl"
        rows = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
            {"id": 3, "text": "test"},
        ]

        write_jsonl(rows, output_path)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed == rows[i]


class TestSampleDataset:
    """Tests for sample_dataset orchestration function."""

    def test_sample_dataset_creates_jsonl_in_split_dir(self, tmp_path):
        """Create sample.jsonl in the split directory."""
        # Setup: Create split directory with mock data
        data_dir = tmp_path / "data"
        split_dir = data_dir / "test/dataset" / "train"
        split_dir.mkdir(parents=True)

        config = {
            "data_dir": str(data_dir),
            "dataset_name": "test/dataset",
            "split": "train",
            "sample_size": 5,
        }

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.select = MagicMock(return_value=[
            {"id": i, "text": f"row_{i}"} for i in range(5)
        ])

        with patch("src.dataset_sampler.load_split", return_value=mock_dataset):
            result_path = sample_dataset(config)

        expected_path = split_dir / "sample.jsonl"
        assert result_path == expected_path
        assert expected_path.exists()

    def test_sample_dataset_fails_on_missing_dataset(self, tmp_path):
        """Raise SamplerError when dataset doesn't exist."""
        config = {
            "data_dir": str(tmp_path / "nonexistent"),
            "dataset_name": "missing/dataset",
            "split": "train",
            "sample_size": 10,
        }

        with pytest.raises(SamplerError) as exc_info:
            sample_dataset(config)

        assert "not found" in str(exc_info.value).lower()
