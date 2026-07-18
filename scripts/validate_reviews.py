"""Validate completed Aquarium duplicate/source review integrity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.review import validate_review_integrity  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hard-fail incomplete or inconsistent reviews.")
    parser.add_argument("records", type=Path, help="Validation image_records.csv.")
    parser.add_argument("duplicates", type=Path, help="Final duplicate review CSV.")
    parser.add_argument("sources", type=Path, help="Final source-group review CSV.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        errors = validate_review_integrity(
            args.records, args.duplicates, args.sources, args.dataset_root
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Review integrity validation failed: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"REVIEW: {error}", file=sys.stderr)
        return 1
    print("Review integrity validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
