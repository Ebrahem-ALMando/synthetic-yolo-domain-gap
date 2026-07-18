"""Generate tables and figures from frozen real manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.split_audit import generate_split_audit  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a frozen real-data split.")
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("bounding_boxes", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/split_audit"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        statistics = generate_split_audit(
            args.manifest_dir, args.bounding_boxes, args.output
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Split audit failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Split audit complete: {statistics['total_images']} images, "
        f"{statistics['total_objects']} objects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
