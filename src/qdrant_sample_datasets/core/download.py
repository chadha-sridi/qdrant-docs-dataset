from pathlib import Path

import httpx


def download_file(url: str, target: Path) -> None:
    """Download `url` to `target` unless it already exists. A partial download never replaces the target."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for data in response.iter_bytes():
                handle.write(data)
    partial.rename(target)
