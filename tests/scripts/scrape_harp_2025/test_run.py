"""Tests for scripts/scrape_harp_2025/run.py - Entry point script."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.config import ConfigError


class TestScriptRun:
    """Tests for the scrape_harp_2025 script."""

    def test_script_loads_config_from_same_directory(self, tmp_path):
        """Verify YAML auto-loaded from script directory."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        config_data = {
            "output_dir": str(tmp_path / "data"),
            "output_filename": "HARP_2025.jsonl",
            "contests": [{"name": "AMC_8", "problems": 25}],
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        from scripts.scrape_harp_2025.run import get_config_path

        with patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir):
            result = get_config_path()
            assert result == config_file

    def test_script_handles_missing_config(self, tmp_path):
        """Handle missing config file gracefully."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        from scripts.scrape_harp_2025.run import main

        with patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "not found" in str(exc_info.value).lower()

    def test_script_runs_with_404_response(self, tmp_path):
        """Script should handle 404 responses gracefully."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        output_dir = tmp_path / "data"
        config_data = {
            "output_dir": str(output_dir),
            "output_filename": "HARP_2025.jsonl",
            "request_delay_seconds": 0,
            "contests": [{"name": "AMC_8", "problems": 1}],
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Mock 404 response
        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir),
            patch("src.harp_scraper.requests.get", return_value=mock_response),
            patch("src.harp_scraper.time.sleep"),
        ):
            from scripts.scrape_harp_2025.run import main

            main()

        # Output file should be created (possibly empty)
        assert (output_dir / "HARP_2025.jsonl").exists()

    def test_script_saves_scraped_problems(self, tmp_path):
        """Script should save scraped problems to JSONL."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        output_dir = tmp_path / "data"
        config_data = {
            "output_dir": str(output_dir),
            "output_filename": "HARP_2025.jsonl",
            "request_delay_seconds": 0,
            "contests": [{"name": "AMC_8", "problems": 1}],
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Mock successful response with problem HTML (matching AoPS wiki format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = r"""
>Problem<
<p>What is $2 + 2$?
$\textbf{(A)}\ 3\qquad\textbf{(B)}\ 4\qquad\textbf{(C)}\ 5\qquad\textbf{(D)}\ 6\qquad\textbf{(E)}\ 7$
</p>
        """

        with (
            patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir),
            patch("src.harp_scraper.requests.get", return_value=mock_response),
            patch("src.harp_scraper.time.sleep"),
        ):
            from scripts.scrape_harp_2025.run import main

            main()

        output_file = output_dir / "HARP_2025.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        problem = json.loads(lines[0])
        assert problem["year"] == "2025"
        assert problem["contest"] == "AMC_8"
        assert problem["number"] == 1

    def test_script_creates_output_directory(self, tmp_path):
        """Script should create output directory if it doesn't exist."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        # Use nested output directory
        output_dir = tmp_path / "nested" / "output" / "dir"
        config_data = {
            "output_dir": str(output_dir),
            "output_filename": "HARP_2025.jsonl",
            "request_delay_seconds": 0,
            "contests": [{"name": "AMC_8", "problems": 1}],
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir),
            patch("src.harp_scraper.requests.get", return_value=mock_response),
            patch("src.harp_scraper.time.sleep"),
        ):
            from scripts.scrape_harp_2025.run import main

            main()

        assert output_dir.exists()
        assert (output_dir / "HARP_2025.jsonl").exists()

    def test_script_validates_config(self, tmp_path):
        """Script should validate config before scraping."""
        script_dir = tmp_path / "scripts" / "scrape_harp_2025"
        script_dir.mkdir(parents=True)

        # Missing required fields
        config_data = {
            "output_dir": str(tmp_path / "data"),
            # Missing output_filename and contests
        }
        config_file = script_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        from scripts.scrape_harp_2025.run import main

        with patch("scripts.scrape_harp_2025.run.SCRIPT_DIR", script_dir):
            with pytest.raises(ConfigError) as exc_info:
                main()

            assert "output_filename" in str(exc_info.value).lower()
