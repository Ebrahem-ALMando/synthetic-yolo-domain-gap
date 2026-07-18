"""Validate a local YOLO dataset and write traceable issue records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.validation import (  # noqa: E402, I001
    validate_yolo_dataset,
    write_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly validate a YOLO detection export.")
    parser.add_argument(
        "dataset_root", type=Path, help="Root containing images, labels, and data.yaml."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_audit/aquarium"),
        help="Directory for machine-readable validation outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        classes, records, issues = validate_yolo_dataset(args.dataset_root)
        write_validation_outputs(args.output, classes, records, issues)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    included = sum(record.inclusion_status == "included" for record in records)
    print(f"Inspected {len(records)} images; included {included}; issues {len(issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
