"""Build conservative source-group proposals and actual-image contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.source_groups import (  # noqa: E402, I001
    generate_duplicate_contact_sheets,
    propose_source_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose evidence-limited source groups; do not freeze a split."
    )
    parser.add_argument("records", type=Path, help="Validated image_records.csv.")
    parser.add_argument("duplicates", type=Path, help="Duplicate candidate CSV.")
    parser.add_argument("dataset_root", type=Path, help="Root containing the acquired images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/reviewed_source_groups.csv"),
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/source_contact_sheets"),
    )
    parser.add_argument(
        "--duplicate-contact-dir",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/duplicate_contact_sheets"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        duplicate_sheets = generate_duplicate_contact_sheets(
            args.duplicates, args.dataset_root, args.duplicate_contact_dir
        )
        summary = propose_source_groups(
            args.records,
            args.duplicates,
            args.dataset_root,
            args.output,
            args.contact_dir,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Source-group proposal failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Duplicate contact-sheet groups: {len(duplicate_sheets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

