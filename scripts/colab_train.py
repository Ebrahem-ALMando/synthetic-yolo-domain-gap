"""Colab entry point for validated single-regime or sequential controlled training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--regime", default="all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-copy", type=Path)
    parser.add_argument("--install-runtime", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != args.expected_revision:
        raise SystemExit(
            f"Repository revision mismatch: expected {args.expected_revision}, got {revision}"
        )
    if args.install_runtime:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "configs/training/runtime-lock.txt",
            ],
            cwd=root,
            check=True,
        )
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("PyTorch did not detect a CUDA GPU; refusing requested GPU training.")
    if not (root / "datasets/experiments/aquarium/v1/real_only/data.yaml").is_file():
        subprocess.run(
            [sys.executable, "scripts/build_experiments.py", "--materialize-only"],
            cwd=root,
            check=True,
        )
    subprocess.run([sys.executable, "scripts/validate_experiments.py"], cwd=root, check=True)
    script = "scripts/run_all_regimes.py" if args.regime == "all" else "scripts/train_yolo.py"
    command = [sys.executable, script]
    if args.regime != "all":
        command.append(args.regime)
    command.extend(["--mode", "final", "--device", args.device, "--confirm-final"])
    result = subprocess.run(command, cwd=root, check=False)
    if args.output_copy and result.returncode == 0:
        import shutil

        destination = args.output_copy.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            root / "artifacts/experiments/final",
            destination / "final",
            dirs_exist_ok=False,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
