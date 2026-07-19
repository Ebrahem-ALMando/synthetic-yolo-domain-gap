"""Validate the Sprint 4B notebook structure and test-set protection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.notebook import validate_notebook  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        type=Path,
        nargs="?",
        default=Path("notebooks/sprint4b_full_training_colab.ipynb"),
    )
    args = parser.parse_args()
    try:
        print(json.dumps(validate_notebook(PROJECT_ROOT / args.notebook), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
