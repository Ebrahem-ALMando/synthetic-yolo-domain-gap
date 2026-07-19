"""Build the actual checksummed, secret-free Sprint 4B CUDA training bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.bundle import (  # noqa: E402
    build_bundle,
    clean_source_state,
    create_inventory,
    required_bundle_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_bundle/aquarium-sprint4b-v2.zip"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = PROJECT_ROOT / args.output
        if args.dry_run:
            source = clean_source_state(PROJECT_ROOT)
            inventory = create_inventory(
                PROJECT_ROOT, required_bundle_files(PROJECT_ROOT), source
            )
        else:
            inventory = build_bundle(PROJECT_ROOT, output)
        print(
            json.dumps(
                {
                    "archive": None
                    if args.dry_run
                    else output.relative_to(PROJECT_ROOT).as_posix(),
                    "archive_sha256": inventory.get("archive_sha256"),
                    "bundle_identity": inventory["bundle_identity"],
                    "expected_repository_revision": inventory["expected_repository_revision"],
                    "source_branch": inventory["source_branch"],
                    "file_count": inventory["file_count"],
                    "total_bytes": inventory["total_bytes"],
                    "source_worktree_dirty": inventory["source_worktree_dirty"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
