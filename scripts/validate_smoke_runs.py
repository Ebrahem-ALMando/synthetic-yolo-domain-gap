"""Validate and summarize the latest completed Sprint 4A smoke run for every regime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/experiments/smoke"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training_environment/sprint4a_smoke_summary.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runs_root = PROJECT_ROOT / args.runs
    completed: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for regime in REGIMES:
        candidates: list[tuple[Path, dict[str, object]]] = []
        for metadata_path in runs_root.glob(f"{regime}-*/run_metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                candidates.append((metadata_path, metadata))
        if not candidates:
            errors.append(f"No completed smoke run exists for {regime}")
            continue
        metadata_path, metadata = max(candidates, key=lambda item: str(item[1]["ended_at_utc"]))
        run_root = metadata_path.parent
        expected = [
            run_root / "resolved_training_config.yaml",
            run_root / "ultralytics/results.csv",
            run_root / "ultralytics/weights/best.pt",
            run_root / "ultralytics/weights/last.pt",
        ]
        missing = [path.name for path in expected if not path.is_file()]
        if missing:
            errors.append(f"{regime} is missing outputs: {', '.join(missing)}")
        if metadata.get("test_set_used") is not False:
            errors.append(f"{regime} does not explicitly record test-set non-use")
        model = metadata.get("model", {})
        if not model.get("pretrained_weight_sha256"):
            errors.append(f"{regime} does not record the pretrained weight hash")
        completed[regime] = {
            "status": metadata["status"],
            "run_directory": run_root.relative_to(PROJECT_ROOT).as_posix(),
            "duration_seconds": metadata["duration_seconds"],
            "process_peak_working_set_bytes": metadata.get("process_peak_working_set_bytes"),
            "process_final_resident_bytes": metadata.get("process_final_resident_bytes"),
            "device": metadata["resolved_arguments"]["device"],
            "image_size": metadata["resolved_arguments"]["imgsz"],
            "batch_size": metadata["resolved_arguments"]["batch"],
            "epoch_count": metadata["resolved_arguments"]["epochs"],
            "regime_manifest_hash": metadata["regime_manifest_hash"],
            "pretrained_weight_sha256": model.get("pretrained_weight_sha256"),
            "test_set_used": metadata["test_set_used"],
            "scientific_result": metadata["scientific_result"],
        }
    summary = {
        "status": "passed" if not errors else "failed",
        "completed_regime_count": len(completed),
        "runs": completed,
        "errors": errors,
        "note": "One-epoch smoke outputs are technical validation only, not scientific results.",
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print("[ERROR] " + "\n[ERROR] ".join(errors), file=sys.stderr)
        return 1
    print("Five-regime smoke gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
