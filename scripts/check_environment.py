"""Run the repository environment checks without requiring package installation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.environment import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
