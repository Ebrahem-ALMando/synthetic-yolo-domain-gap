"""Generate data-derived statistics and visual sheets for Aquarium synthetic data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.audit import generate_synthetic_audit  # noqa: E402
from synthdet.synthetic.contracts import load_synthetic_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic/aquarium_synthetic_v1.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_synthetic_config(PROJECT_ROOT / args.config)
        logical_full = Path(config.dataset["output_directory"])
        logical_output = Path(args.output or logical_full)
        bank_manifests = PROJECT_ROOT / Path(config.dataset["manifest_directory"])
        statistics = generate_synthetic_audit(
            config,
            PROJECT_ROOT,
            PROJECT_ROOT / logical_output,
            logical_output,
            PROJECT_ROOT / logical_full,
            bank_manifests,
            PROJECT_ROOT / Path(args.manifests or config.dataset["manifest_directory"]),
            PROJECT_ROOT / Path(args.report or config.dataset["report_directory"]),
        )
    except (FileNotFoundError, KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        print(f"Synthetic audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(statistics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
