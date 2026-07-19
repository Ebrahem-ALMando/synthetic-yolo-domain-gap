"""Run a validated YOLO smoke or final regime configuration without test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.runner import run_training  # noqa: E402

REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regime", choices=REGIMES)
    parser.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required safety acknowledgement for a final training run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "final" and not args.confirm_final:
        print("[ERROR] Final mode requires --confirm-final.", file=sys.stderr)
        return 2
    regime_path = PROJECT_ROOT / f"configs/training/regimes/{args.regime}.yaml"
    try:
        result = run_training(
            regime_path,
            PROJECT_ROOT / "configs/training/common.yaml",
            PROJECT_ROOT / "manifests/aquarium/experiments/v1/experiment_metadata.json",
            PROJECT_ROOT,
            args.mode,
            args.device,
            [sys.executable, *sys.argv],
            profile_path=args.profile,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
