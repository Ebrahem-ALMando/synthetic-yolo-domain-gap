"""Reproducible, credential-safe Roboflow export acquisition."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthdet.data.hashing import sha256_file

ROBOFLOW_API_ROOT = "https://api.roboflow.com"


def export_api_url(workspace: str, project: str, version: int, export_format: str) -> str:
    if not workspace or not project or not export_format or version <= 0:
        raise ValueError("workspace, project, format, and a positive version are required")
    safe_parts = [
        urllib.parse.quote(part, safe="")
        for part in (workspace, project, str(version), export_format)
    ]
    return f"{ROBOFLOW_API_ROOT}/{'/'.join(safe_parts)}"


def _ensure_empty_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Raw destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        archive.extractall(destination)


def _make_read_only(destination: Path) -> None:
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IREAD)


def import_archive(
    archive_path: Path,
    destination: Path,
    source_url: str,
    acquisition_method: str,
) -> dict[str, Any]:
    """Copy and extract an acquired ZIP into an immutable raw tree."""

    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Not a readable ZIP archive: {archive_path}")
    _ensure_empty_destination(destination)
    stored_archive = destination / "source_export.zip"
    shutil.copy2(archive_path, stored_archive)
    _safe_extract(stored_archive, destination / "export")
    metadata: dict[str, Any] = {
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "acquisition_method": acquisition_method,
        "source_url": source_url,
        "archive_filename": stored_archive.name,
        "archive_sha256": sha256_file(stored_archive),
        "raw_files_are_immutable": True,
    }
    (destination / "acquisition_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _make_read_only(destination)
    return metadata


def acquire_roboflow_export(
    destination: Path,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Request an official export link, download it, and preserve the raw ZIP."""

    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for automatic Roboflow acquisition")
    endpoint = export_api_url(workspace, project, version, export_format)
    request_url = endpoint + "?" + urllib.parse.urlencode({"api_key": key})
    with urllib.request.urlopen(request_url, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    download_url = payload.get("export", {}).get("link")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError("Roboflow did not return a valid HTTPS export link")
    with tempfile.TemporaryDirectory(prefix="synthdet-aquarium-") as temporary:
        archive_path = Path(temporary) / "aquarium.zip"
        with (
            urllib.request.urlopen(download_url, timeout=120) as response,  # noqa: S310
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        return import_archive(
            archive_path,
            destination,
            source_url=endpoint,
            acquisition_method="roboflow_api",
        )
