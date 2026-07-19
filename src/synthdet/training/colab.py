"""Restart-safe state and persistent-copy helpers for external CUDA training."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")


def resolve_expected_revision(root: Path, override: str | None = None) -> str:
    inventory = root / "training_bundle_inventory.json"
    if not inventory.is_file():
        raise FileNotFoundError("Validated internal training-bundle inventory is required")
    metadata = json.loads(inventory.read_text(encoding="utf-8"))
    revision = metadata.get("expected_repository_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Bundle inventory has no valid expected_repository_revision")
    if metadata.get("source_branch") != "main":
        raise ValueError("Bundle inventory source branch is not main")
    if metadata.get("source_worktree_dirty") is not False:
        raise ValueError("Bundle inventory records a dirty source worktree")
    if override is not None and override != revision:
        raise ValueError("Explicit revision override conflicts with bundle inventory")
    return revision


def load_state(path: Path, expected_revision: str, profile_identity: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": 1,
            "expected_repository_revision": expected_revision,
            "hardware_profile_identity": profile_identity,
            "test_set_access_count": 0,
            "regimes": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("expected_repository_revision") != expected_revision:
        raise ValueError("Completion state belongs to a different repository revision")
    if state.get("hardware_profile_identity") != profile_identity:
        raise ValueError("Completion state belongs to a different frozen hardware profile")
    if int(state.get("test_set_access_count", -1)) != 0:
        raise ValueError("Completion state does not prove zero test-set access")
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_state_record(saved: dict[str, Any], validated: dict[str, Any]) -> None:
    if saved.get("status") != "completed" or int(saved.get("test_set_access_count", -1)) != 0:
        raise ValueError("Saved regime state is not a zero-test-access completion")
    for key in ("best_pt_sha256", "last_pt_sha256", "results_csv_sha256"):
        if saved.get(key) != validated.get(key):
            raise ValueError(f"Persistent completed-run hash differs from state: {key}")


def start_attempt(state: dict[str, Any], regime: str) -> None:
    previous = state["regimes"].get(regime)
    history = list(previous.get("attempts", [])) if previous else []
    if previous:
        preserved = {key: value for key, value in previous.items() if key != "attempts"}
        if preserved.get("status") == "running":
            preserved["status"] = "interrupted"
            preserved["recovered_at_utc"] = datetime.now(UTC).isoformat()
        history.append(preserved)
    state["regimes"][regime] = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "attempts": history,
    }


def persist_run(source: Path, destination_root: Path) -> Path:
    """Copy then atomically publish a run; partial copies never look completed."""

    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    staging = destination_root / f".{source.name}.copying"
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite persistent run: {destination}")
    shutil.copytree(source, staging)
    staging.replace(destination)
    return destination
