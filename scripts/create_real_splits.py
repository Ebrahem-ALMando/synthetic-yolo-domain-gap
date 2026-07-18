"""Freeze deterministic group-aware real train/validation/test manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.splitting import create_real_splits  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create immutable 70/20/10 real manifests using seed 42."
    )
    parser.add_argument("records", type=Path, help="Validation image_records.csv.")
    parser.add_argument("duplicates", type=Path, help="Reviewed duplicate-candidate CSV.")
    parser.add_argument("--source-groups", type=Path, help="CSV: image_path,source_group_id.")
    parser.add_argument("--output", type=Path, default=Path("manifests/aquarium"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-unknown-source-groups",
        action="store_true",
        help="Use singleton fallback only after explicitly documenting unknown provenance.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = create_real_splits(
            args.records,
            args.duplicates,
            args.output,
            source_groups_path=args.source_groups,
            seed=args.seed,
            allow_unknown_source_groups=args.allow_unknown_source_groups,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Split creation failed: {error}", file=sys.stderr)
        return 1
    print(f"Frozen split identity: {metadata['combined_split_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

