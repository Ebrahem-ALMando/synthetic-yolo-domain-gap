"""Create exact and perceptual duplicate candidates for human review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.duplicates import analyze_duplicates  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group SHA-256 matches and dHash near-duplicate candidates; delete nothing."
    )
    parser.add_argument("records", type=Path, help="Validation image_records.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/duplicate_candidates.csv"),
    )
    parser.add_argument(
        "--threshold", type=int, default=6, help="Maximum 64-bit dHash Hamming distance."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        count = analyze_duplicates(args.records, args.output, args.threshold)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Duplicate analysis failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {count} candidate duplicate groups; no files were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

