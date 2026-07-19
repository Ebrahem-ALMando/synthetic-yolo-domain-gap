"""Validate the frozen or smoke Aquarium synthetic dataset and provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import load_synthetic_config, verify_active_split  # noqa: E402
from synthdet.synthetic.validation import (  # noqa: E402
    validate_object_bank,
    validate_synthetic_pool,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic/aquarium_synthetic_v1.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests", type=Path)
    parser.add_argument("--count", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_synthetic_config(PROJECT_ROOT / args.config)
        verify_active_split(
            PROJECT_ROOT / Path(config.dataset["active_split_directory"]),
            config.split_identity,
        )
        full_output = PROJECT_ROOT / Path(config.dataset["output_directory"])
        bank_manifests = PROJECT_ROOT / Path(config.dataset["manifest_directory"])
        logical_output = Path(args.output or config.dataset["output_directory"])
        manifests = PROJECT_ROOT / Path(args.manifests or config.dataset["manifest_directory"])
        errors = validate_object_bank(config, PROJECT_ROOT, full_output, bank_manifests)
        errors.extend(
            validate_synthetic_pool(
                config,
                PROJECT_ROOT,
                PROJECT_ROOT / logical_output,
                logical_output,
                manifests,
                bank_manifests,
                args.count or int(config.dataset["full_pool_size"]),
            )
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"Synthetic validation failed: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print("Synthetic dataset and object-bank validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
