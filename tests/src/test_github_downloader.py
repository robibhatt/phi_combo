"""Tests for src/github_downloader.py - GitHub dataset download logic."""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.github_downloader import (
    HarpDownloadError,
    download_and_extract_zip,
    download_harp_dataset,
    validate_harp_config,
)


class TestValidateHarpConfig:
    """Tests for validate_harp_config function."""

    def test_valid_config_passes(self):
        """Valid configuration should not raise."""
        config = {
            "data_dir": "/path/to/data",
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        validate_harp_config(config)  # Should not raise

    def test_missing_data_dir_raises(self):
        """Missing data_dir should raise HarpDownloadError."""
        config = {
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "data_dir" in str(exc_info.value)

    def test_missing_dataset_name_raises(self):
        """Missing dataset_name should raise HarpDownloadError."""
        config = {
            "data_dir": "/path/to/data",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "dataset_name" in str(exc_info.value)

    def test_missing_files_raises(self):
        """Missing files should raise HarpDownloadError."""
        config = {
            "data_dir": "/path/to/data",
            "dataset_name": "HARP",
            "base_url": "https://example.com",
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "files" in str(exc_info.value)

    def test_missing_base_url_raises(self):
        """Missing base_url should raise HarpDownloadError."""
        config = {
            "data_dir": "/path/to/data",
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "base_url" in str(exc_info.value)

    def test_files_not_list_raises(self):
        """files must be a list."""
        config = {
            "data_dir": "/path/to/data",
            "dataset_name": "HARP",
            "files": "HARP.jsonl.zip",  # Should be a list
            "base_url": "https://example.com",
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "list" in str(exc_info.value)

    def test_empty_files_list_raises(self):
        """Empty files list should raise HarpDownloadError."""
        config = {
            "data_dir": "/path/to/data",
            "dataset_name": "HARP",
            "files": [],
            "base_url": "https://example.com",
        }
        with pytest.raises(HarpDownloadError) as exc_info:
            validate_harp_config(config)
        assert "empty" in str(exc_info.value).lower()


class TestDownloadAndExtractZip:
    """Tests for download_and_extract_zip function."""

    def _create_zip_bytes(self, filename: str, content: str) -> bytes:
        """Helper to create a ZIP file in memory."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, content)
        return buffer.getvalue()

    def test_download_and_extract_success(self, tmp_path):
        """Successfully download and extract a ZIP file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        zip_content = self._create_zip_bytes("test.jsonl", '{"key": "value"}\n')
        mock_response = MagicMock()
        mock_response.content = zip_content
        mock_response.raise_for_status = MagicMock()

        with patch("src.github_downloader.requests.get", return_value=mock_response):
            result = download_and_extract_zip(
                "https://example.com/test.jsonl.zip", output_dir
            )

        assert result == output_dir / "test.jsonl"
        assert (output_dir / "test.jsonl").exists()
        assert (output_dir / "test.jsonl").read_text() == '{"key": "value"}\n'

    def test_download_network_error_raises(self, tmp_path):
        """Network error should raise HarpDownloadError."""
        import requests

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch(
            "src.github_downloader.requests.get",
            side_effect=requests.RequestException("Network error"),
        ):
            with pytest.raises(HarpDownloadError) as exc_info:
                download_and_extract_zip(
                    "https://example.com/test.jsonl.zip", output_dir
                )
            assert "download" in str(exc_info.value).lower()

    def test_invalid_zip_raises(self, tmp_path):
        """Invalid ZIP content should raise HarpDownloadError."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_response = MagicMock()
        mock_response.content = b"not a zip file"
        mock_response.raise_for_status = MagicMock()

        with patch("src.github_downloader.requests.get", return_value=mock_response):
            with pytest.raises(HarpDownloadError) as exc_info:
                download_and_extract_zip(
                    "https://example.com/test.jsonl.zip", output_dir
                )
            assert "zip" in str(exc_info.value).lower()

    def test_creates_output_dir_if_not_exists(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "output"

        zip_content = self._create_zip_bytes("test.jsonl", '{"key": "value"}\n')
        mock_response = MagicMock()
        mock_response.content = zip_content
        mock_response.raise_for_status = MagicMock()

        with patch("src.github_downloader.requests.get", return_value=mock_response):
            download_and_extract_zip("https://example.com/test.jsonl.zip", output_dir)

        assert output_dir.exists()
        assert (output_dir / "test.jsonl").exists()


class TestDownloadHarpDataset:
    """Tests for download_harp_dataset function."""

    def _create_zip_bytes(self, filename: str, content: str) -> bytes:
        """Helper to create a ZIP file in memory."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, content)
        return buffer.getvalue()

    def test_download_all_files(self, tmp_path):
        """Download all specified files from config."""
        data_dir = tmp_path / "data"
        config = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip", "HARP_mcq.jsonl.zip"],
            "base_url": "https://example.com",
        }

        def mock_get(url, **kwargs):
            if "HARP.jsonl.zip" in url:
                content = self._create_zip_bytes("HARP.jsonl", '{"id": 1}\n')
            else:
                content = self._create_zip_bytes("HARP_mcq.jsonl", '{"id": 2}\n')
            mock_response = MagicMock()
            mock_response.content = content
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("src.github_downloader.requests.get", side_effect=mock_get):
            result = download_harp_dataset(config)

        assert "HARP.jsonl.zip" in result
        assert "HARP_mcq.jsonl.zip" in result
        assert (data_dir / "HARP" / "HARP.jsonl").exists()
        assert (data_dir / "HARP" / "HARP_mcq.jsonl").exists()

    def test_skip_existing_files(self, tmp_path):
        """Skip download if extracted file already exists."""
        data_dir = tmp_path / "data"
        dataset_dir = data_dir / "HARP"
        dataset_dir.mkdir(parents=True)

        # Create existing file
        existing_file = dataset_dir / "HARP.jsonl"
        existing_file.write_text('{"existing": true}\n')

        config = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }

        with patch("src.github_downloader.requests.get") as mock_get:
            result = download_harp_dataset(config)
            mock_get.assert_not_called()

        # Original file should be unchanged
        assert existing_file.read_text() == '{"existing": true}\n'
        assert result["HARP.jsonl.zip"] == existing_file

    def test_returns_paths_dict(self, tmp_path):
        """Return dictionary mapping filenames to paths."""
        data_dir = tmp_path / "data"
        config = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }

        zip_content = self._create_zip_bytes("HARP.jsonl", '{"id": 1}\n')
        mock_response = MagicMock()
        mock_response.content = zip_content
        mock_response.raise_for_status = MagicMock()

        with patch("src.github_downloader.requests.get", return_value=mock_response):
            result = download_harp_dataset(config)

        assert isinstance(result, dict)
        assert "HARP.jsonl.zip" in result
        assert isinstance(result["HARP.jsonl.zip"], Path)

    def test_validates_config_before_download(self, tmp_path):
        """Config validation should happen before any download."""
        config = {
            "data_dir": str(tmp_path),
            # Missing required fields
        }

        with patch("src.github_downloader.requests.get") as mock_get:
            with pytest.raises(HarpDownloadError):
                download_harp_dataset(config)
            mock_get.assert_not_called()

    def test_constructs_correct_url(self, tmp_path):
        """URL should be constructed from base_url and filename."""
        data_dir = tmp_path / "data"
        config = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://raw.githubusercontent.com/aadityasingh/HARP/main",
        }

        zip_content = self._create_zip_bytes("HARP.jsonl", '{"id": 1}\n')
        mock_response = MagicMock()
        mock_response.content = zip_content
        mock_response.raise_for_status = MagicMock()

        with patch(
            "src.github_downloader.requests.get", return_value=mock_response
        ) as mock_get:
            download_harp_dataset(config)
            mock_get.assert_called_once_with(
                "https://raw.githubusercontent.com/aadityasingh/HARP/main/HARP.jsonl.zip",
                timeout=300,
            )
