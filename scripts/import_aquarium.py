"""Validate raw Aquarium data and copy only included files to a working tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.validation import (  # noqa: E402, I001
    normalize_included_records,
    validate_yolo_dataset,
    write_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a normalized working copy without changing immutable raw data."
    )
    parser.add_argument("dataset_root", type=Path, help="Acquired raw YOLO export root.")
    parser.add_argument(
        "--working-root", type=Path, default=Path("datasets/working/aquarium")
    )
    parser.add_argument(
        "--report-root", type=Path, default=Path("reports/dataset_audit/aquarium")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        classes, records, issues = validate_yolo_dataset(args.dataset_root)
        write_validation_outputs(args.report_root, classes, records, issues)
        copied = normalize_included_records(args.dataset_root, args.working_root, records)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1
    print(f"Copied {copied} validated image/label pairs; raw files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

