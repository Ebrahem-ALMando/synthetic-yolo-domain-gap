"""Run all five controlled regimes sequentially in smoke or explicitly confirmed final mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-final", action="store_true")
    parser.add_argument("--profile", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "final" and not args.confirm_final:
        print("[ERROR] Final mode requires --confirm-final.", file=sys.stderr)
        return 2
    for regime in REGIMES:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_yolo.py"),
            regime,
            "--mode",
            args.mode,
            "--device",
            args.device,
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.confirm_final:
            command.append("--confirm-final")
        if args.profile:
            command.extend(["--profile", str(args.profile)])
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
