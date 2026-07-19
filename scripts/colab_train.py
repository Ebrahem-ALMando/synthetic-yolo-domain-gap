"""Run five frozen regimes sequentially with validated immediate persistence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.colab import (  # noqa: E402
    REGIMES,
    load_state,
    persist_run,
    read_source_revision,
    start_attempt,
    validate_state_record,
    write_state,
)
from synthdet.training.completion import load_frozen_profile, validate_completed_run  # noqa: E402
from synthdet.training.cuda import inspect_cuda  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--regime", choices=("all", *REGIMES), default="all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--persistent-output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository.resolve()
    try:
        if read_source_revision(root) != args.expected_revision:
            raise ValueError("Repository/bundle revision differs from --expected-revision")
        inspect_cuda(args.device, root)
        common = yaml.safe_load((root / "configs/training/common.yaml").read_text())
        profile = load_frozen_profile(args.profile.resolve(), common)
        if str(profile["device"]) != str(args.device):
            raise ValueError("Requested CUDA device differs from frozen profile")
        metadata = json.loads(
            (root / "manifests/aquarium/experiments/v1/experiment_metadata.json").read_text()
        )
        persistent = args.persistent_output.resolve()
        state_path = persistent / "completion_state.json"
        state = load_state(state_path, args.expected_revision, profile["profile_identity"])
        selected = REGIMES if args.regime == "all" else (args.regime,)
        local_root = root / common["outputs"]["run_root"] / "final"
        for regime in selected:
            expected_hash = metadata["regime_manifest_hashes"][f"{regime}.csv"]
            previous = state["regimes"].get(regime)
            if previous and previous.get("status") == "completed":
                persistent_run = Path(previous["persistent_run_directory"])
                validated = validate_completed_run(
                    persistent_run, regime, common, profile, expected_hash
                )
                validate_state_record(previous, validated)
                print(f"[SKIP] {regime}: persistent completed run revalidated")
                continue
            before = set(local_root.glob(f"{regime}-*")) if local_root.is_dir() else set()
            start_attempt(state, regime)
            write_state(state_path, state)
            command = [
                sys.executable,
                "scripts/train_yolo.py",
                regime,
                "--mode",
                "final",
                "--device",
                args.device,
                "--confirm-final",
                "--profile",
                str(args.profile.resolve()),
            ]
            try:
                subprocess.run(command, cwd=root, check=True)
                after = set(local_root.glob(f"{regime}-*"))
                created = sorted(after - before)
                if len(created) != 1:
                    raise RuntimeError(f"Expected one new {regime} run, found {len(created)}")
                validate_completed_run(created[0], regime, common, profile, expected_hash)
                copied = persist_run(created[0], persistent / "runs")
                persisted = validate_completed_run(copied, regime, common, profile, expected_hash)
                state["regimes"][regime] = {
                    "status": "completed",
                    "attempts": state["regimes"][regime]["attempts"],
                    "local_run_directory": str(created[0]),
                    "persistent_run_directory": str(copied),
                    "best_pt_sha256": persisted["best_pt_sha256"],
                    "last_pt_sha256": persisted["last_pt_sha256"],
                    "results_csv_sha256": persisted["results_csv_sha256"],
                    "ended_at_utc": persisted["ended_at_utc"],
                    "test_set_access_count": 0,
                }
                write_state(state_path, state)
                print(f"[PERSISTED] {regime}: {copied}")
            except BaseException as error:
                state["regimes"][regime] = {
                    **state["regimes"][regime],
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "ended_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                write_state(state_path, state)
                raise
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
