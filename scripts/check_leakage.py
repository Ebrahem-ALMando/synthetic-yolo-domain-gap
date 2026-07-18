"""Hard-fail validation for real splits and future synthetic-source manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.leakage import validate_leakage  # noqa: E402, I001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail on path, hash, duplicate, or source leakage."
    )
    parser.add_argument(
        "manifest_dir", type=Path, help="Directory containing frozen real manifests."
    )
    parser.add_argument(
        "--synthetic-source-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--synthetic-background-manifest", type=Path, action="append", default=[]
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        errors = validate_leakage(
            args.manifest_dir,
            args.synthetic_source_manifest,
            args.synthetic_background_manifest,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Leakage validation failed: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"LEAKAGE: {error}", file=sys.stderr)
        return 1
    print("Leakage validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
