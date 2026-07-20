"""Validate sealed Sprint 5 outputs and generate deterministic post-campaign analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.evaluation.analysis import generate_error_analysis  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(generate_error_analysis(PROJECT_ROOT), indent=2))
