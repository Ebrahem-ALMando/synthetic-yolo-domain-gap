"""Record the local Sprint 4A training hardware and software environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.environment import (  # noqa: E402
    collect_environment,
    write_environment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=Path("reports/training_environment/sprint4a.json")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/training_environment/sprint4a.md")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = collect_environment(PROJECT_ROOT)
    write_environment_report(report, PROJECT_ROOT / args.json, PROJECT_ROOT / args.summary)
    print(f"Training environment: {report['classification']}")
    print(f"PyTorch CUDA available: {report['cuda_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
