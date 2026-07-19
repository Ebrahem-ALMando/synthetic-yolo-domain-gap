"""Run all five final-training dry-runs through the actual entry point without CUDA training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import stable_json_hash  # noqa: E402
from synthdet.training.colab import REGIMES, resolve_expected_revision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    return parser


def _validation_profile(device: str) -> dict[str, object]:
    common = yaml.safe_load((PROJECT_ROOT / "configs/training/common.yaml").read_text())
    weight = yaml.safe_load((PROJECT_ROOT / "configs/training/base_weight.yaml").read_text())
    preflight = [
        {
            "profile_name": "standard",
            "batch": 16,
            "imgsz": 640,
            "status": "bundle_runtime_validation_only",
            "scientific_result": False,
        }
    ]
    identity_inputs = {
        "profile_name": "standard",
        "batch": 16,
        "imgsz": 640,
        "device": str(device),
        "base_weight_sha256": weight["sha256"],
        "identities": common["identities"],
        "preflight": preflight,
    }
    return {
        **identity_inputs,
        "status": "frozen",
        "environment": {
            "gpu_model": "bundle-runtime-validation-only",
            "software_versions": {},
        },
        "profile_identity": stable_json_hash(identity_inputs),
        "scientific_result": False,
        "test_set_access_count": 0,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        revision = resolve_expected_revision(PROJECT_ROOT)
        results: list[dict[str, str]] = []
        with tempfile.TemporaryDirectory(prefix="synthdet-runtime-profile-") as temporary:
            profile_path = Path(temporary) / "final_profile.json"
            profile_path.write_text(
                json.dumps(_validation_profile(args.device), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for regime in REGIMES:
                command = [
                    sys.executable,
                    "scripts/train_yolo.py",
                    regime,
                    "--mode",
                    "final",
                    "--device",
                    args.device,
                    "--confirm-final",
                    "--dry-run",
                    "--profile",
                    str(profile_path),
                ]
                completed = subprocess.run(
                    command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
                )
                runtime_validated = '"status": "dry_run_validated"' in completed.stdout
                if completed.returncode != 0 or not runtime_validated:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    raise RuntimeError(f"{regime} final runtime dry-run failed: {detail}")
                results.append({"regime": regime, "status": "dry_run_validated"})
        print(
            json.dumps(
                {
                    "status": "passed",
                    "expected_repository_revision": revision,
                    "final_runtime_dry_runs": results,
                    "test_set_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
