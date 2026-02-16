"""Tests for src/dataset_loader.py - Dataset download logic."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.dataset_loader import get_data_dir, dataset_exists, download_dataset


class TestGetDataDir:
    """Tests for get_data_dir function."""

    def test_data_dir_created_if_not_exists(self, tmp_path):
        """Verify directory creation when missing."""
        data_dir = tmp_path / "data" / "hugging_face"
        config = {"data_dir": str(data_dir)}

        result = get_data_dir(config)

        assert result == data_dir
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_data_dir_returns_existing_directory(self, tmp_path):
        """Return existing directory without error."""
        data_dir = tmp_path / "data" / "hugging_face"
        data_dir.mkdir(parents=True)
        config = {"data_dir": str(data_dir)}

        result = get_data_dir(config)

        assert result == data_dir
        assert data_dir.exists()

    def test_data_dir_resolves_relative_path(self, tmp_path, monkeypatch):
        """Resolve relative paths from project root."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        config = {"data_dir": "../data/hugging_face"}

        result = get_data_dir(config)

        expected = (project_root / "../data/hugging_face").resolve()
        assert result == expected
        assert expected.exists()


class TestDatasetExists:
    """Tests for dataset_exists function."""

    def test_dataset_exists_returns_true_when_present(self, tmp_path):
        """Check detection of existing dataset."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a dataset directory with some content
        dataset_dir = data_dir / "imdb"
        dataset_dir.mkdir()
        (dataset_dir / "dataset_info.json").touch()

        result = dataset_exists(data_dir, "imdb")

        assert result is True

    def test_dataset_exists_returns_false_when_missing(self, tmp_path):
        """Check detection of missing dataset."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = dataset_exists(data_dir, "imdb")

        assert result is False

    def test_dataset_exists_returns_false_for_empty_dir(self, tmp_path):
        """Empty dataset directory counts as not existing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dataset_dir = data_dir / "imdb"
        dataset_dir.mkdir()  # Empty directory

        result = dataset_exists(data_dir, "imdb")

        assert result is False


class TestDownloadDataset:
    """Tests for download_dataset function."""

    def test_download_skipped_when_dataset_exists(self, tmp_path):
        """Verify no download when data present."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create existing dataset
        dataset_dir = data_dir / "imdb"
        dataset_dir.mkdir()
        (dataset_dir / "dataset_info.json").touch()

        config = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "imdb", "subset": None}]
        }

        with patch("src.dataset_loader.datasets.load_dataset") as mock_load:
            download_dataset(config)
            mock_load.assert_not_called()

    def test_download_triggered_when_dataset_missing(self, tmp_path):
        """Verify download when data absent."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "imdb", "subset": None}]
        }

        mock_dataset = MagicMock()
        with patch("src.dataset_loader.datasets.load_dataset", return_value=mock_dataset) as mock_load:
            download_dataset(config)
            mock_load.assert_called_once()
            mock_dataset.save_to_disk.assert_called_once()

    def test_dataset_stored_in_correct_location(self, tmp_path):
        """Verify correct storage path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "imdb", "subset": None}]
        }

        mock_dataset = MagicMock()
        with patch("src.dataset_loader.datasets.load_dataset", return_value=mock_dataset):
            download_dataset(config)
            expected_path = data_dir / "imdb"
            mock_dataset.save_to_disk.assert_called_once_with(str(expected_path))

    def test_download_with_subset(self, tmp_path):
        """Download dataset with specific subset."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "data_dir": str(data_dir),
            "datasets": [{"name": "squad", "subset": "plain_text"}]
        }

        mock_dataset = MagicMock()
        with patch("src.dataset_loader.datasets.load_dataset", return_value=mock_dataset) as mock_load:
            download_dataset(config)
            mock_load.assert_called_once_with("squad", "plain_text")

    def test_download_multiple_datasets(self, tmp_path):
        """Download multiple datasets from config."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "data_dir": str(data_dir),
            "datasets": [
                {"name": "imdb", "subset": None},
                {"name": "squad", "subset": "plain_text"}
            ]
        }

        mock_dataset = MagicMock()
        with patch("src.dataset_loader.datasets.load_dataset", return_value=mock_dataset) as mock_load:
            download_dataset(config)
            assert mock_load.call_count == 2
            assert mock_dataset.save_to_disk.call_count == 2
