"""Freeze deterministic group-aware real train/validation/test manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.splitting import (  # noqa: E402, I001
    create_real_splits,
    preview_real_splits,
    verify_real_split_reproduction,
)


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
        "--dataset-root",
        type=Path,
        required=True,
        help="Raw export root used for mandatory review path-integrity validation.",
    )
    parser.add_argument(
        "--allow-unknown-source-groups",
        action="store_true",
        help="Use singleton fallback only after explicitly documenting unknown provenance.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Compute a disposable candidate split without freezing output files.",
    )
    mode.add_argument(
        "--verify-frozen",
        action="store_true",
        help="Reproduce in temporary storage and compare with --output without overwriting it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        options = {
            "records_path": args.records,
            "duplicates_path": args.duplicates,
            "source_groups_path": args.source_groups,
            "seed": args.seed,
            "allow_unknown_source_groups": args.allow_unknown_source_groups,
            "dataset_root": args.dataset_root,
        }
        if args.preview:
            metadata = preview_real_splits(**options)
        elif args.verify_frozen:
            metadata = verify_real_split_reproduction(args.output, **options)
        else:
            metadata = create_real_splits(output_dir=args.output, **options)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Split creation failed: {error}", file=sys.stderr)
        return 1
    label = "Reproduced" if args.verify_frozen else "Candidate" if args.preview else "Frozen"
    print(f"{label} split identity: {metadata['combined_split_sha256']}")
    print(f"Image counts: {metadata['actual_counts']}")
    print(f"Class image counts: {metadata['image_count_per_class_per_split']}")
    print(f"Class coverage limitations: {metadata['class_coverage_limitations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
