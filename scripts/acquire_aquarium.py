"""Acquire or safely import the approved Roboflow Aquarium export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.acquisition import (  # noqa: E402, I001
    acquire_roboflow_export,
    export_api_url,
    import_archive,
)

WORKSPACE = "brad-dwyer"
PROJECT = "aquarium-combined"
VERSION = 2
EXPORT_FORMAT = "yolov5pytorch"
SOURCE_PAGE = "https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire Aquarium Combined v2 without modifying source files in place."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download", action="store_true", help="Download using ROBOFLOW_API_KEY.")
    action.add_argument("--archive", type=Path, help="Import a manually downloaded Roboflow ZIP.")
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and print the credential-safe plan.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("datasets/raw/aquarium"),
        help="Immutable raw-data destination (default: datasets/raw/aquarium).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    endpoint = export_api_url(WORKSPACE, PROJECT, VERSION, EXPORT_FORMAT)
    if args.dry_run:
        print(f"Dataset page: {SOURCE_PAGE}")
        print(f"Export API endpoint: {endpoint}")
        print(f"Destination: {args.destination}")
        print("Automatic download requires environment variable ROBOFLOW_API_KEY.")
        print("No network request or filesystem change was made.")
        return 0
    try:
        if args.archive:
            metadata = import_archive(
                args.archive,
                args.destination,
                source_url=SOURCE_PAGE,
                acquisition_method="manual_roboflow_export",
            )
        else:
            metadata = acquire_roboflow_export(
                args.destination, WORKSPACE, PROJECT, VERSION, EXPORT_FORMAT
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Acquisition failed: {error}", file=sys.stderr)
        return 1
    print(f"Raw export acquired with SHA-256 {metadata['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
