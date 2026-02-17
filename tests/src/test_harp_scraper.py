"""Tests for src/harp_scraper.py - HARP 2025 dataset scraper."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.harp_scraper import (
    HarpScraperError,
    build_url,
    get_2025_contests,
    has_choices,
    replace_img_alt_text,
    standardize_boxed_command,
    replace_unicode_chars,
    parse_html_to_text,
    find_closing_brace,
    extract_choices,
    extract_last_boxed_from_text,
    map_difficulty,
    map_subject,
    fetch_problem_page,
    scrape_single_problem,
    scrape_all_2025_problems,
    save_to_jsonl,
)


class TestHarpScraperError:
    """Tests for HarpScraperError exception."""

    def test_error_is_exception(self):
        """HarpScraperError should be an Exception."""
        assert issubclass(HarpScraperError, Exception)

    def test_error_message(self):
        """Error should preserve message."""
        error = HarpScraperError("test error")
        assert str(error) == "test error"


class TestBuildUrl:
    """Tests for build_url function."""

    def test_build_amc_url(self):
        """Build URL for AMC problems."""
        url = build_url(2025, "AMC_10A", 1)
        assert url == "https://artofproblemsolving.com/wiki/index.php/2025_AMC_10A_Problems/Problem_1"

    def test_build_aime_url(self):
        """Build URL for AIME problems."""
        url = build_url(2025, "AIME_I", 15)
        assert url == "https://artofproblemsolving.com/wiki/index.php/2025_AIME_I_Problems/Problem_15"

    def test_build_amc_8_url(self):
        """Build URL for AMC 8 problems."""
        url = build_url(2025, "AMC_8", 10)
        assert url == "https://artofproblemsolving.com/wiki/index.php/2025_AMC_8_Problems/Problem_10"


class TestGet2025Contests:
    """Tests for get_2025_contests function."""

    def test_returns_list(self):
        """get_2025_contests should return a list."""
        contests = get_2025_contests()
        assert isinstance(contests, list)

    def test_contains_all_contests(self):
        """Should contain all 2025 AMC/AIME contests."""
        contests = get_2025_contests()
        contest_names = [c["name"] for c in contests]
        expected = ["AMC_8", "AMC_10A", "AMC_10B", "AMC_12A", "AMC_12B", "AIME_I", "AIME_II"]
        for name in expected:
            assert name in contest_names

    def test_contest_has_required_fields(self):
        """Each contest should have name and problems fields."""
        contests = get_2025_contests()
        for contest in contests:
            assert "name" in contest
            assert "problems" in contest
            assert isinstance(contest["problems"], int)


class TestHasChoices:
    """Tests for has_choices function."""

    def test_amc_has_choices(self):
        """AMC contests should have choices."""
        assert has_choices("AMC_8") is True
        assert has_choices("AMC_10A") is True
        assert has_choices("AMC_10B") is True
        assert has_choices("AMC_12A") is True
        assert has_choices("AMC_12B") is True

    def test_aime_has_no_choices(self):
        """AIME contests should not have choices."""
        assert has_choices("AIME_I") is False
        assert has_choices("AIME_II") is False


class TestReplaceImgAltText:
    """Tests for replace_img_alt_text function."""

    def test_replace_latex_img(self):
        """Extract LaTeX from image alt text."""
        html = '<img src="foo.png" alt="$x^2$" />'
        result = replace_img_alt_text(html)
        assert result == "$x^2$"

    def test_drop_image_filename(self):
        """Drop alt text that looks like an image filename."""
        html = '<img src="foo.png" alt="diagram.png" />'
        result = replace_img_alt_text(html)
        assert result == ""

    def test_preserve_aligned_latex(self):
        """Preserve LaTeX with \\begin{...} blocks."""
        html = '<img alt="\\begin{align}x=1\\end{align}" />'
        result = replace_img_alt_text(html)
        assert "\\begin{align}" in result


class TestStandardizeBoxedCommand:
    """Tests for standardize_boxed_command function."""

    def test_replace_fbox(self):
        """Replace \\fbox with \\boxed."""
        text = "Answer is \\fbox{42}"
        result = standardize_boxed_command(text)
        assert "\\boxed{42}" in result

    def test_replace_framebox(self):
        """Replace \\framebox with \\boxed."""
        text = "Answer is \\framebox[l]{42}"
        result = standardize_boxed_command(text)
        assert "\\boxed{42}" in result

    def test_fix_boxed_space(self):
        """Fix space between \\boxed and brace."""
        text = "Answer is \\boxed {42}"
        result = standardize_boxed_command(text)
        assert "\\boxed{42}" in result


class TestReplaceUnicodeChars:
    """Tests for replace_unicode_chars function."""

    def test_replace_nbsp(self):
        """Replace non-breaking space."""
        text = "hello\u00a0world"
        result = replace_unicode_chars(text)
        assert result == "hello world"

    def test_replace_smart_quotes(self):
        """Replace smart quotes with standard quotes."""
        text = "\u201chello\u201d"
        result = replace_unicode_chars(text)
        assert result == "'hello'"


class TestFindClosingBrace:
    """Tests for find_closing_brace function."""

    def test_simple_brace(self):
        """Find closing brace in simple case."""
        text = "42}"
        assert find_closing_brace(text) == 2

    def test_nested_braces(self):
        """Handle nested braces."""
        text = "{inner}outer}"
        assert find_closing_brace(text) == 12

    def test_no_closing_brace(self):
        """Return -1 if no closing brace found."""
        text = "no closing brace"
        assert find_closing_brace(text) == -1


class TestExtractLastBoxedFromText:
    """Tests for extract_last_boxed_from_text function."""

    def test_extract_simple_boxed(self):
        """Extract content from \\boxed{...}."""
        text = "The answer is $\\boxed{42}$."
        result = extract_last_boxed_from_text(text)
        assert result == "42"

    def test_extract_last_when_multiple(self):
        """Extract from last \\boxed when multiple present."""
        text = "$\\boxed{A}$ then $\\boxed{B}$"
        result = extract_last_boxed_from_text(text)
        assert result == "B"

    def test_no_boxed_returns_none(self):
        """Return None if no \\boxed present."""
        text = "No boxed answer here"
        result = extract_last_boxed_from_text(text)
        assert result is None


class TestExtractChoices:
    """Tests for extract_choices function."""

    def test_extract_standard_choices(self):
        """Extract choices from standard format."""
        text = r"$\textbf{(A)}\ 1\qquad\textbf{(B)}\ 2\qquad\textbf{(C)}\ 3\qquad\textbf{(D)}\ 4\qquad\textbf{(E)}\ 5$"
        result = extract_choices(text)
        assert result is not None
        assert "choices" in result
        assert "A" in result["choices"]
        assert "E" in result["choices"]

    def test_no_choices_returns_none(self):
        """Return None if no choices found."""
        text = "This is a problem without choices."
        result = extract_choices(text)
        assert result is None


class TestMapDifficulty:
    """Tests for map_difficulty function."""

    def test_amc8_difficulty(self):
        """AMC 8 difficulty mapping."""
        assert map_difficulty("2025", "AMC_8", 1) == 1
        assert map_difficulty("2025", "AMC_8", 20) == 1
        assert map_difficulty("2025", "AMC_8", 25) == 2

    def test_amc10_difficulty(self):
        """AMC 10 difficulty mapping."""
        assert map_difficulty("2025", "AMC_10A", 1) == 1
        assert map_difficulty("2025", "AMC_10A", 10) == 2
        assert map_difficulty("2025", "AMC_10A", 25) == 3

    def test_amc12_difficulty(self):
        """AMC 12 difficulty mapping."""
        assert map_difficulty("2025", "AMC_12A", 5) == 2
        assert map_difficulty("2025", "AMC_12A", 15) == 3
        assert map_difficulty("2025", "AMC_12A", 25) == 4

    def test_aime_difficulty(self):
        """AIME difficulty mapping."""
        assert map_difficulty("2025", "AIME_I", 1) == 3
        assert map_difficulty("2025", "AIME_I", 7) == 4
        assert map_difficulty("2025", "AIME_I", 11) == 5
        assert map_difficulty("2025", "AIME_I", 15) == 6


class TestMapSubject:
    """Tests for map_subject function."""

    def test_amc8_subject(self):
        """AMC 8 should map to prealgebra."""
        assert map_subject("AMC_8") == "prealgebra"

    def test_amc10_subject(self):
        """AMC 10 should map to algebra."""
        assert map_subject("AMC_10A") == "algebra"
        assert map_subject("AMC_10B") == "algebra"

    def test_amc12_subject(self):
        """AMC 12 should map to algebra."""
        assert map_subject("AMC_12A") == "algebra"
        assert map_subject("AMC_12B") == "algebra"

    def test_aime_subject(self):
        """AIME should map to algebra."""
        assert map_subject("AIME_I") == "algebra"
        assert map_subject("AIME_II") == "algebra"


class TestFetchProblemPage:
    """Tests for fetch_problem_page function."""

    def test_successful_fetch(self):
        """Successfully fetch a page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>content</html>"

        with patch("src.harp_scraper.requests.get", return_value=mock_response):
            result = fetch_problem_page("https://example.com", MagicMock())
            assert result.text == "<html>content</html>"

    def test_404_returns_none(self):
        """Return None for 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("src.harp_scraper.requests.get", return_value=mock_response):
            result = fetch_problem_page("https://example.com", MagicMock())
            assert result is None

    def test_retry_on_error(self):
        """Retry on network error."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.harp_scraper.requests.get", side_effect=[
            requests.RequestException("error"),
            mock_response,
        ]):
            with patch("src.harp_scraper.time.sleep"):
                result = fetch_problem_page(
                    "https://example.com",
                    MagicMock(),
                    retry_count=2,
                    retry_backoff=0,
                )
                assert result is not None


class TestScrapeAllProblems:
    """Tests for scrape_all_2025_problems function."""

    def test_returns_tuple_with_list_and_stats(self):
        """Should return a tuple of (list of problems, stats dict)."""
        config = {
            "output_dir": "/tmp/test",
            "output_filename": "test.jsonl",
            "contests": [{"name": "AMC_8", "problems": 1}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 404  # Simulate missing page

        with patch("src.harp_scraper.requests.get", return_value=mock_response):
            with patch("src.harp_scraper.time.sleep"):
                result, stats = scrape_all_2025_problems(config)
                assert isinstance(result, list)
                assert isinstance(stats, dict)
                assert "found" in stats
                assert "skipped" in stats
                assert "not_found" in stats
                assert "errors" in stats


class TestSaveToJsonl:
    """Tests for save_to_jsonl function."""

    def test_save_creates_file(self, tmp_path):
        """Create JSONL file with problems."""
        problems = [
            {"year": "2025", "contest": "AMC_8", "number": 1, "problem": "Test"},
        ]
        output_path = tmp_path / "output.jsonl"

        save_to_jsonl(problems, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["problem"] == "Test"

    def test_save_empty_list(self, tmp_path):
        """Save empty list creates empty file."""
        output_path = tmp_path / "output.jsonl"

        save_to_jsonl([], output_path)

        assert output_path.exists()
        assert output_path.read_text() == ""

    def test_save_creates_parent_dirs(self, tmp_path):
        """Create parent directories if they don't exist."""
        problems = [{"year": "2025", "contest": "AMC_8", "number": 1}]
        output_path = tmp_path / "nested" / "dirs" / "output.jsonl"

        save_to_jsonl(problems, output_path)

        assert output_path.exists()
