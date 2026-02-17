"""Tests for scripts/download_harp/run.py - Entry point script."""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.github_downloader import HarpDownloadError


class TestScriptRun:
    """Tests for the download_harp script."""

    def test_script_loads_config_from_same_directory(self, tmp_path):
        """Verify YAML auto-loaded from script directory."""
        script_dir = tmp_path / "scripts" / "download_harp"
        script_dir.mkdir(parents=True)

        config_data = {
            "data_dir": str(tmp_path / "data"),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        from scripts.download_harp.run import get_config_path

        with patch("scripts.download_harp.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == config_file

    def test_script_runs_end_to_end(self, tmp_path):
        """Integration test for full workflow."""
        script_dir = tmp_path / "scripts" / "download_harp"
        script_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        config_data = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Create mock ZIP response
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("HARP.jsonl", '{"id": 1}\n')
        zip_content = buffer.getvalue()

        mock_response = MagicMock()
        mock_response.content = zip_content
        mock_response.raise_for_status = MagicMock()

        with patch("scripts.download_harp.run.SCRIPT_DIR", script_dir), patch(
            "src.github_downloader.requests.get", return_value=mock_response
        ):
            from scripts.download_harp.run import main

            main()

        # Verify file was extracted
        assert (data_dir / "HARP" / "HARP.jsonl").exists()

    def test_script_handles_missing_config(self, tmp_path):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "download_harp"
        script_dir.mkdir(parents=True)

        from scripts.download_harp.run import main
        from src.config import ConfigError

        with patch("scripts.download_harp.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_skips_existing_files(self, tmp_path):
        """Skip download when file already exists."""
        script_dir = tmp_path / "scripts" / "download_harp"
        script_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        dataset_dir = data_dir / "HARP"
        dataset_dir.mkdir(parents=True)

        # Create existing file
        (dataset_dir / "HARP.jsonl").write_text('{"existing": true}\n')

        config_data = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("scripts.download_harp.run.SCRIPT_DIR", script_dir), patch(
            "src.github_downloader.requests.get"
        ) as mock_get:
            from scripts.download_harp.run import main

            main()

            # Should not download since file exists
            mock_get.assert_not_called()

    def test_script_handles_download_error(self, tmp_path):
        """Handle download errors gracefully."""
        import requests

        script_dir = tmp_path / "scripts" / "download_harp"
        script_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        config_data = {
            "data_dir": str(data_dir),
            "dataset_name": "HARP",
            "files": ["HARP.jsonl.zip"],
            "base_url": "https://example.com",
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("scripts.download_harp.run.SCRIPT_DIR", script_dir), patch(
            "src.github_downloader.requests.get",
            side_effect=requests.RequestException("Network error"),
        ):
            from scripts.download_harp.run import main

            with pytest.raises(HarpDownloadError):
                main()
