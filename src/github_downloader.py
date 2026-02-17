"""GitHub dataset download logic for ZIP-compressed JSONL files."""

import io
import zipfile
from pathlib import Path
from typing import Any

import requests


class HarpDownloadError(Exception):
    """Raised when HARP dataset download fails."""

    pass


def validate_harp_config(config: dict[str, Any]) -> None:
    """Validate HARP download configuration.

    Args:
        config: Configuration dictionary.

    Raises:
        HarpDownloadError: If required fields are missing or invalid.
    """
    required_fields = ["data_dir", "dataset_name", "files", "base_url"]

    for field in required_fields:
        if field not in config:
            raise HarpDownloadError(f"Missing required field: '{field}'")

    if not isinstance(config["files"], list):
        raise HarpDownloadError("'files' must be a list")

    if len(config["files"]) == 0:
        raise HarpDownloadError("'files' list cannot be empty")


def download_and_extract_zip(url: str, output_dir: Path) -> Path:
    """Download a ZIP file and extract its contents.

    Args:
        url: URL to download the ZIP file from.
        output_dir: Directory to extract files to.

    Returns:
        Path to the extracted file.

    Raises:
        HarpDownloadError: If download or extraction fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HarpDownloadError(f"Failed to download {url}: {e}")

    try:
        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            # Get the list of files in the ZIP
            file_list = zf.namelist()
            if not file_list:
                raise HarpDownloadError(f"ZIP file from {url} is empty")

            # Extract all files
            zf.extractall(output_dir)

            # Return path to the first extracted file
            extracted_path = output_dir / file_list[0]
            return extracted_path
    except zipfile.BadZipFile:
        raise HarpDownloadError(f"Invalid ZIP file from {url}")


def download_harp_dataset(config: dict[str, Any]) -> dict[str, Path]:
    """Download HARP dataset files from GitHub.

    Args:
        config: Configuration dictionary with:
            - data_dir: Base data directory
            - dataset_name: Name of the dataset (e.g., "HARP")
            - files: List of ZIP files to download
            - base_url: Base URL for downloads

    Returns:
        Dictionary mapping filenames to their extracted paths.

    Raises:
        HarpDownloadError: If validation or download fails.
    """
    validate_harp_config(config)

    data_dir = Path(config["data_dir"]).resolve()
    dataset_name = config["dataset_name"]
    base_url = config["base_url"].rstrip("/")
    files = config["files"]

    output_dir = data_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    for zip_filename in files:
        # Determine expected extracted filename (remove .zip extension)
        if zip_filename.endswith(".zip"):
            extracted_filename = zip_filename[:-4]
        else:
            extracted_filename = zip_filename

        extracted_path = output_dir / extracted_filename

        # Skip if already exists
        if extracted_path.exists():
            print(f"File '{extracted_filename}' already exists, skipping download.")
            results[zip_filename] = extracted_path
            continue

        # Download and extract
        url = f"{base_url}/{zip_filename}"
        print(f"Downloading {zip_filename}...")

        extracted_path = download_and_extract_zip(url, output_dir)
        print(f"Extracted to {extracted_path}")

        results[zip_filename] = extracted_path

    return results
