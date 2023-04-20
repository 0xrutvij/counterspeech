from pathlib import Path

import requests


def download_file(url: str, path: Path) -> Path:
    """Download a file from a URL to a given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return path
