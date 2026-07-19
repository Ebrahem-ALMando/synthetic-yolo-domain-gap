"""Build the immutable train-only object bank for Aquarium synthetic V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import load_synthetic_config, verify_active_split  # noqa: E402
from synthdet.synthetic.object_bank import build_object_bank  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic/aquarium_synthetic_v1.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_synthetic_config(PROJECT_ROOT / args.config)
        split_dir = PROJECT_ROOT / Path(config.dataset["active_split_directory"])
        verify_active_split(split_dir, config.split_identity)
        logical_output = Path(config.dataset["output_directory"])
        output = PROJECT_ROOT / (args.output or logical_output)
        manifests = PROJECT_ROOT / (args.manifests or Path(config.dataset["manifest_directory"]))
        summary = build_object_bank(config, PROJECT_ROOT, output, logical_output, manifests)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"Object-bank creation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
