"""Validate the frozen Sprint 5 contract and generate its tracked JSON input record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.evaluation.contract import validate_contract, write_input_contract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/sprint5_final.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/evaluation/sprint5_input_contract.json",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Validate without rewriting the JSON record."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract = args.contract.resolve()
        validation = validate_contract(PROJECT_ROOT, contract)
        if not args.check_only:
            write_input_contract(PROJECT_ROOT, contract, args.output.resolve(), validation)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
