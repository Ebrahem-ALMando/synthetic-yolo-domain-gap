"""Generate factual dataset statistics from completed validation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.audit import generate_dataset_audit  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an audit from inspected image records.")
    parser.add_argument(
        "validation_dir", type=Path, help="Directory produced by validate_dataset.py."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/dataset_audit/aquarium")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        statistics = generate_dataset_audit(args.validation_dir, args.output)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Audit generated from {statistics['inspected_images']} inspected images; "
        f"{statistics['included_images']} included."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

