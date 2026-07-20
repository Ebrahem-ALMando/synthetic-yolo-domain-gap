"""Run the single locked Sprint 5 protected-test evaluation campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.evaluation.campaign import run_campaign  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen, complete five-model Sprint 5 evaluation exactly once."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to the committed frozen Sprint 5 YAML contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_campaign(PROJECT_ROOT, (PROJECT_ROOT / args.contract).resolve())
    print(
        json.dumps(
            {
                "status": result["lock"]["status"],
                "campaign_id": result["lock"]["campaign_id"],
                "recommended_model": result["recommended_model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
