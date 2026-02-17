"""Tests for src/dataset_filter.py - Dataset filtering by year logic."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.dataset_filter import (
    FilterError,
    extract_year_from_link,
    categorize_by_year,
    filter_dataset,
    write_splits,
    discover_splits,
    filter_and_split_dataset,
)


class TestExtractYearFromLink:
    """Tests for extract_year_from_link function."""

    def test_extracts_year_from_standard_aops_link(self):
        """Extract year from standard AOPS wiki link."""
        link = "https://artofproblemsolving.com/wiki/index.php/2004_AIME_II_Problems/Problem_1"
        result = extract_year_from_link(link)
        assert result == 2004

    def test_extracts_year_from_amc_link(self):
        """Extract year from AMC link."""
        link = "https://artofproblemsolving.com/wiki/index.php/1983_AMC_8_Problems/Problem_5"
        result = extract_year_from_link(link)
        assert result == 1983

    def test_extracts_year_from_usamo_link(self):
        """Extract year from USAMO link."""
        link = "https://artofproblemsolving.com/wiki/index.php/2015_USAMO_Problems/Problem_3"
        result = extract_year_from_link(link)
        assert result == 2015

    def test_returns_none_for_invalid_link(self):
        """Return None when no year found in link."""
        link = "https://example.com/no-year-here"
        result = extract_year_from_link(link)
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Return None for empty string input."""
        result = extract_year_from_link("")
        assert result is None

    def test_returns_none_for_none_input(self):
        """Return None when input is None."""
        result = extract_year_from_link(None)
        assert result is None

    def test_extracts_first_year_from_link_with_multiple_numbers(self):
        """Extract first valid year pattern from complex links."""
        link = "https://artofproblemsolving.com/wiki/index.php/2010_AMC_10A_Problems/Problem_15"
        result = extract_year_from_link(link)
        assert result == 2010


class TestCategorizeByYear:
    """Tests for categorize_by_year function."""

    def test_year_2023_goes_to_train(self):
        """Year 2023 (train_max) should go to train split."""
        result = categorize_by_year(
            year=2023, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "train"

    def test_year_2020_goes_to_train(self):
        """Year before train_max should go to train split."""
        result = categorize_by_year(
            year=2020, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "train"

    def test_year_2024_goes_to_valid(self):
        """Year matching valid_year should go to valid split."""
        result = categorize_by_year(
            year=2024, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "valid"

    def test_year_2025_goes_to_test(self):
        """Year at test_min should go to test split."""
        result = categorize_by_year(
            year=2025, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "test"

    def test_year_2030_goes_to_test(self):
        """Year after test_min should go to test split."""
        result = categorize_by_year(
            year=2030, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "test"

    def test_none_year_goes_to_unknown(self):
        """None year should go to unknown split."""
        result = categorize_by_year(
            year=None, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "unknown"

    def test_very_old_year_goes_to_train(self):
        """Very old year (1950) should go to train split."""
        result = categorize_by_year(
            year=1950, train_max=2023, valid_year=2024, test_min=2025
        )
        assert result == "train"


class TestFilterDataset:
    """Tests for filter_dataset function."""

    def test_splits_dataset_into_correct_buckets(self):
        """Dataset rows are split into correct buckets based on year."""
        # Mock dataset with mixed years
        mock_rows = [
            {"link": "https://aops.com/2020_AMC_Problems/1", "problem": "p1"},
            {"link": "https://aops.com/2024_AMC_Problems/2", "problem": "p2"},
            {"link": "https://aops.com/2025_AMC_Problems/3", "problem": "p3"},
            {"link": "https://aops.com/no_year", "problem": "p4"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=len(mock_rows))

        year_boundaries = {
            "train_max_year": 2023,
            "valid_year": 2024,
            "test_min_year": 2025,
        }

        result = filter_dataset(mock_dataset, year_boundaries)

        assert len(result["train"]) == 1
        assert len(result["valid"]) == 1
        assert len(result["test"]) == 1
        assert len(result["unknown"]) == 1

    def test_handles_empty_dataset(self):
        """Empty dataset returns empty splits."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_dataset.__len__ = MagicMock(return_value=0)

        year_boundaries = {
            "train_max_year": 2023,
            "valid_year": 2024,
            "test_min_year": 2025,
        }

        result = filter_dataset(mock_dataset, year_boundaries)

        assert result["train"] == []
        assert result["valid"] == []
        assert result["test"] == []
        assert result["unknown"] == []

    def test_preserves_all_original_fields(self):
        """All original fields are preserved in output."""
        mock_rows = [
            {
                "link": "https://aops.com/2020_AMC_Problems/1",
                "letter": "A",
                "answer": "42",
                "problem": "What is 6*7?",
                "solution": "6*7=42",
            }
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=1)

        year_boundaries = {
            "train_max_year": 2023,
            "valid_year": 2024,
            "test_min_year": 2025,
        }

        result = filter_dataset(mock_dataset, year_boundaries)

        row = result["train"][0]
        assert row["link"] == "https://aops.com/2020_AMC_Problems/1"
        assert row["letter"] == "A"
        assert row["answer"] == "42"
        assert row["problem"] == "What is 6*7?"
        assert row["solution"] == "6*7=42"

    def test_adds_extracted_year_metadata(self):
        """Each row has _extracted_year field added."""
        mock_rows = [
            {"link": "https://aops.com/2020_AMC_Problems/1", "problem": "p1"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=1)

        year_boundaries = {
            "train_max_year": 2023,
            "valid_year": 2024,
            "test_min_year": 2025,
        }

        result = filter_dataset(mock_dataset, year_boundaries)

        row = result["train"][0]
        assert "_extracted_year" in row
        assert row["_extracted_year"] == 2020

    def test_all_train_years_go_to_train(self):
        """All rows with year <= train_max go to train."""
        mock_rows = [
            {"link": "https://aops.com/2000_AMC_Problems/1", "problem": "p1"},
            {"link": "https://aops.com/2010_AMC_Problems/2", "problem": "p2"},
            {"link": "https://aops.com/2023_AMC_Problems/3", "problem": "p3"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=3)

        year_boundaries = {
            "train_max_year": 2023,
            "valid_year": 2024,
            "test_min_year": 2025,
        }

        result = filter_dataset(mock_dataset, year_boundaries)

        assert len(result["train"]) == 3
        assert len(result["valid"]) == 0
        assert len(result["test"]) == 0


class TestWriteSplits:
    """Tests for write_splits function."""

    def test_writes_jsonl_files_for_each_split(self, tmp_path):
        """Create JSONL files for train, valid, test, unknown."""
        splits = {
            "train": [{"problem": "p1", "_extracted_year": 2020}],
            "valid": [{"problem": "p2", "_extracted_year": 2024}],
            "test": [{"problem": "p3", "_extracted_year": 2025}],
            "unknown": [{"problem": "p4", "_extracted_year": None}],
        }

        result = write_splits(splits, tmp_path)

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "valid.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()
        assert (tmp_path / "unknown.jsonl").exists()

        assert result["train"] == tmp_path / "train.jsonl"
        assert result["valid"] == tmp_path / "valid.jsonl"
        assert result["test"] == tmp_path / "test.jsonl"
        assert result["unknown"] == tmp_path / "unknown.jsonl"

    def test_writes_valid_jsonl_format(self, tmp_path):
        """Each file contains valid JSONL format."""
        splits = {
            "train": [
                {"problem": "p1", "_extracted_year": 2020},
                {"problem": "p2", "_extracted_year": 2021},
            ],
            "valid": [],
            "test": [],
            "unknown": [],
        }

        write_splits(splits, tmp_path)

        train_path = tmp_path / "train.jsonl"
        lines = train_path.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert "problem" in parsed
            assert "_extracted_year" in parsed

    def test_empty_splits_create_empty_files(self, tmp_path):
        """Empty splits create empty JSONL files."""
        splits = {
            "train": [{"problem": "p1", "_extracted_year": 2020}],
            "valid": [],
            "test": [],
            "unknown": [],
        }

        write_splits(splits, tmp_path)

        assert (tmp_path / "valid.jsonl").exists()
        assert (tmp_path / "valid.jsonl").read_text() == ""

    def test_creates_output_directory_if_not_exists(self, tmp_path):
        """Create output directory structure if it doesn't exist."""
        output_dir = tmp_path / "nested" / "output" / "dir"
        splits = {
            "train": [{"problem": "p1", "_extracted_year": 2020}],
            "valid": [],
            "test": [],
            "unknown": [],
        }

        write_splits(splits, output_dir)

        assert output_dir.exists()
        assert (output_dir / "train.jsonl").exists()


class TestDiscoverSplits:
    """Tests for discover_splits function."""

    def test_discovers_single_split(self, tmp_path):
        """Find single split directory."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()

        result = discover_splits(dataset_dir)

        assert result == ["train"]

    def test_discovers_multiple_splits(self, tmp_path):
        """Find multiple split directories."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "test").mkdir()

        result = discover_splits(dataset_dir)

        assert sorted(result) == ["test", "train"]

    def test_raises_error_for_nonexistent_directory(self, tmp_path):
        """Raise FilterError when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FilterError) as exc_info:
            discover_splits(nonexistent)

        assert "not found" in str(exc_info.value).lower()

    def test_raises_error_for_empty_directory(self, tmp_path):
        """Raise FilterError when directory has no splits."""
        dataset_dir = tmp_path / "empty_dataset"
        dataset_dir.mkdir()

        with pytest.raises(FilterError) as exc_info:
            discover_splits(dataset_dir)

        assert "no splits found" in str(exc_info.value).lower()

    def test_ignores_files(self, tmp_path):
        """Only return directories, not files."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "README.md").touch()

        result = discover_splits(dataset_dir)

        assert result == ["train"]


class TestFilterAndSplitDataset:
    """Tests for filter_and_split_dataset orchestration function."""

    def test_end_to_end_filtering(self, tmp_path):
        """Integration test: config -> filtered JSONL files."""
        # Setup input data directory with discoverable split
        dataset_dir = tmp_path / "input" / "di-zhang-fdu" / "AOPS"
        (dataset_dir / "train").mkdir(parents=True)

        # Setup output directory
        output_dir = tmp_path / "output"

        config = {
            "input_data_dir": str(tmp_path / "input"),
            "output_data_dir": str(output_dir),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }

        # Mock dataset
        mock_rows = [
            {"link": "https://aops.com/2020_AMC_Problems/1", "problem": "p1"},
            {"link": "https://aops.com/2024_AMC_Problems/2", "problem": "p2"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=2)

        with patch("src.dataset_filter.load_split", return_value=mock_dataset):
            result = filter_and_split_dataset(config)

        expected_output_dir = output_dir / "di-zhang-fdu" / "AOPS"
        assert result["train"] == expected_output_dir / "train.jsonl"
        assert result["valid"] == expected_output_dir / "valid.jsonl"

        # Verify files exist
        assert (expected_output_dir / "train.jsonl").exists()
        assert (expected_output_dir / "valid.jsonl").exists()

    def test_raises_error_when_input_not_found(self, tmp_path):
        """Raise FilterError when input dataset doesn't exist."""
        config = {
            "input_data_dir": str(tmp_path / "nonexistent"),
            "output_data_dir": str(tmp_path / "output"),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }

        with pytest.raises(FilterError) as exc_info:
            filter_and_split_dataset(config)

        assert "not found" in str(exc_info.value).lower()

    def test_uses_link_field_for_year_extraction(self, tmp_path):
        """Use 'link' field from dataset rows for year extraction."""
        dataset_dir = tmp_path / "input" / "di-zhang-fdu" / "AOPS"
        (dataset_dir / "train").mkdir(parents=True)

        config = {
            "input_data_dir": str(tmp_path / "input"),
            "output_data_dir": str(tmp_path / "output"),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }

        mock_rows = [
            {"link": "https://aops.com/1995_IMO_Problems/1", "problem": "p1"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
        mock_dataset.__len__ = MagicMock(return_value=1)

        with patch("src.dataset_filter.load_split", return_value=mock_dataset):
            filter_and_split_dataset(config)

        output_dir = tmp_path / "output" / "di-zhang-fdu" / "AOPS"
        train_content = (output_dir / "train.jsonl").read_text()
        assert "1995" in train_content

    def test_processes_multiple_input_splits(self, tmp_path):
        """Process multiple input splits and merge into output splits."""
        dataset_dir = tmp_path / "input" / "di-zhang-fdu" / "AOPS"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "test").mkdir(parents=True)

        config = {
            "input_data_dir": str(tmp_path / "input"),
            "output_data_dir": str(tmp_path / "output"),
            "dataset_name": "di-zhang-fdu/AOPS",
            "year_boundaries": {
                "train_max_year": 2023,
                "valid_year": 2024,
                "test_min_year": 2025,
            },
        }

        # Different mock data for each split
        call_count = [0]

        def mock_load_split(path):
            mock_rows = [
                {"link": f"https://aops.com/2020_AMC_Problems/{call_count[0]}", "problem": f"p{call_count[0]}"},
            ]
            call_count[0] += 1
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_rows))
            mock_dataset.__len__ = MagicMock(return_value=1)
            return mock_dataset

        with patch("src.dataset_filter.load_split", side_effect=mock_load_split):
            filter_and_split_dataset(config)

        output_dir = tmp_path / "output" / "di-zhang-fdu" / "AOPS"
        train_lines = (output_dir / "train.jsonl").read_text().strip().split("\n")
        # Should have 2 rows merged from both input splits
        assert len(train_lines) == 2
