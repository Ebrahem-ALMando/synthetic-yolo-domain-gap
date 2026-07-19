from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from synthdet.synthetic.contracts import sha256_file, stable_json_hash, write_csv
from synthdet.training.bundle import build_bundle, safe_extract, validate_extracted_bundle
from synthdet.training.colab import (
    load_state,
    persist_run,
    start_attempt,
    validate_state_record,
    write_state,
)
from synthdet.training.completion import build_results_archive, validate_completed_run
from synthdet.training.cuda import select_common_profile
from synthdet.training.notebook import validate_notebook


def test_bundle_creation_and_archive_checksum(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "payload.txt"
    payload.write_text("frozen", encoding="utf-8")
    entry = {
        "path": "payload.txt",
        "size_bytes": payload.stat().st_size,
        "sha256": sha256_file(payload),
    }
    identity_inputs = {
        "bundle_version": "fixture",
        "expected_repository_revision": "a" * 40,
        "real_split_identity": "r" * 64,
        "synthetic_pool_identity": "s" * 64,
        "experiment_design_identity": "e" * 64,
        "base_weight_sha256": "w" * 64,
        "files": [entry],
    }
    inventory = {
        **identity_inputs,
        "bundle_identity": stable_json_hash(identity_inputs),
        "inventory": [entry],
        "file_count": 1,
        "total_bytes": payload.stat().st_size,
    }
    monkeypatch.setattr("synthdet.training.bundle.required_bundle_files", lambda _: [payload])
    monkeypatch.setattr("synthdet.training.bundle.create_inventory", lambda *_: inventory.copy())
    archive = tmp_path / "bundle.zip"
    result = build_bundle(source, archive)
    assert archive.is_file()
    assert archive.with_suffix(".zip.inventory.json").is_file()
    assert archive.with_suffix(".zip.sha256").read_text().split()[0] == sha256_file(archive)
    assert result["bundle_identity"] == inventory["bundle_identity"]


def _extracted_fixture(root: Path, protected_collision: bool = False) -> None:
    manifest = root / "manifests/aquarium/v2/real_test.csv"
    manifest.parent.mkdir(parents=True)
    protected_hash = "f" * 64
    write_csv(
        manifest,
        ["image_path", "content_hash"],
        [{"image_path": "datasets/protected.jpg", "content_hash": protected_hash}],
    )
    payload = root / ("copy.jpg" if protected_collision else "payload.txt")
    payload.write_text("payload", encoding="utf-8")
    entry = {
        "path": payload.relative_to(root).as_posix(),
        "size_bytes": payload.stat().st_size,
        "sha256": protected_hash if protected_collision else sha256_file(payload),
    }
    inputs = {
        "bundle_version": "fixture",
        "expected_repository_revision": "a" * 40,
        "real_split_identity": "r" * 64,
        "synthetic_pool_identity": "s" * 64,
        "experiment_design_identity": "e" * 64,
        "base_weight_sha256": "w" * 64,
        "files": [entry],
    }
    inventory = {**inputs, "bundle_identity": stable_json_hash(inputs), "inventory": [entry]}
    (root / "training_bundle_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")


def test_bundle_test_image_exclusion_and_safe_extraction(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    _extracted_fixture(valid)
    assert validate_extracted_bundle(valid)["bundle_identity"]
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    _extracted_fixture(invalid, protected_collision=True)
    with pytest.raises(ValueError, match="Protected real-test content"):
        validate_extracted_bundle(invalid)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract(archive, tmp_path / "extract")


def test_notebook_configuration_rendering_and_test_protection() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_notebook(root / "notebooks/sprint4b_full_training_colab.ipynb")
    assert result["section_count"] == 16
    assert result["no_test_inference_commands"] is True


def test_common_profile_selection_rules() -> None:
    def standard_ok(name: str, batch: int) -> dict[str, object]:
        return {
            "profile_name": name,
            "batch": batch,
            "status": "passed",
            "free_vram_bytes_after": 2 * 1024**3,
        }

    assert select_common_profile(standard_ok, 8 * 1024**3)[0] == "standard"

    def fallback(name: str, batch: int) -> dict[str, object]:
        if batch == 16:
            raise RuntimeError("CUDA out of memory")
        return {
            "profile_name": name,
            "batch": batch,
            "status": "passed",
            "free_vram_bytes_after": 1024**3,
        }

    assert select_common_profile(fallback, 8 * 1024**3)[0] == "low_memory"


def test_completion_state_validation_and_interrupted_handling(tmp_path: Path) -> None:
    state_path = tmp_path / "completion_state.json"
    state = load_state(state_path, "a" * 40, "p" * 64)
    state["regimes"]["real_50"] = {"status": "interrupted"}
    write_state(state_path, state)
    loaded = load_state(state_path, "a" * 40, "p" * 64)
    assert loaded["regimes"]["real_50"]["status"] == "interrupted"
    loaded["regimes"]["real_50"] = {"status": "running", "started_at_utc": "before"}
    start_attempt(loaded, "real_50")
    assert loaded["regimes"]["real_50"]["attempts"][0]["status"] == "interrupted"
    with pytest.raises(ValueError, match="different repository"):
        load_state(state_path, "b" * 40, "p" * 64)


def _run_fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    run = root / "real_only-fixture"
    (run / "ultralytics/weights").mkdir(parents=True)
    (run / "ultralytics/weights/best.pt").write_bytes(b"best")
    (run / "ultralytics/weights/last.pt").write_bytes(b"last")
    (run / "ultralytics/results.png").write_bytes(b"figure")
    (run / "ultralytics/results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.5,0.4,0.3,0.2\n",
        encoding="utf-8",
    )
    identities = {"real_split": "r" * 64, "synthetic_pool": "s" * 64, "experiment_design": "e" * 64}
    profile_inputs = {
        "profile_name": "standard",
        "batch": 16,
        "imgsz": 640,
        "device": "0",
        "base_weight_sha256": "w" * 64,
        "identities": identities,
        "preflight": [],
    }
    profile = {
        **profile_inputs,
        "status": "frozen",
        "profile_identity": stable_json_hash(profile_inputs),
        "environment": {"gpu_model": "fixture", "software_versions": {}},
    }
    metadata = {
        "status": "completed",
        "regime": "real_only",
        "mode": "final",
        "test_set_used": False,
        "test_set_access_count": 0,
        "regime_manifest_hash": "m" * 64,
        "identities": identities,
        "hardware_profile": profile,
        "model": {"pretrained_weight_sha256": "w" * 64},
        "git_revision": "a" * 40,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "ended_at_utc": "2026-01-01T00:01:00+00:00",
        "duration_seconds": 60,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "resolved_training_config.yaml").write_text(
        yaml.safe_dump(
            {
                "batch": 16,
                "imgsz": 640,
                "resume": False,
                "optimizer": "AdamW",
                "epochs": 1,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    common = {"identities": identities, "model": {"architecture": "YOLO11n"}}
    return common, profile


def test_completed_run_hash_validation_and_no_test_evidence(tmp_path: Path) -> None:
    common, profile = _run_fixture(tmp_path)
    run = tmp_path / "real_only-fixture"
    record = validate_completed_run(run, "real_only", common, profile, "m" * 64)
    assert record["best_pt_sha256"] == sha256_file(run / "ultralytics/weights/best.pt")
    saved = {**record, "status": "completed", "test_set_access_count": 0}
    validate_state_record(saved, record)
    saved["best_pt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        validate_state_record(saved, record)
    (run / "test_predictions").mkdir()
    with pytest.raises(ValueError, match="prohibited test output"):
        validate_completed_run(run, "real_only", common, profile, "m" * 64)


def test_persistent_copy_and_results_archive_inventory(tmp_path: Path) -> None:
    common, profile = _run_fixture(tmp_path)
    run = tmp_path / "real_only-fixture"
    copied = persist_run(run, tmp_path / "persistent")
    assert copied.is_dir() and not (copied.parent / f".{copied.name}.copying").exists()
    record = validate_completed_run(copied, "real_only", common, profile, "m" * 64)
    completion = tmp_path / "completion.json"
    table = tmp_path / "table.csv"
    completion.write_text("{}", encoding="utf-8")
    table.write_text("notice\nNON-FINAL VALIDATION RESULTS\n", encoding="utf-8")
    manifest = {"combined_sprint4b_training_identity": "t" * 64, "runs": [record]}
    archive = tmp_path / "results.zip"
    inventory = build_results_archive(manifest, archive, completion, table)
    assert inventory["contains_dataset_images"] is False
    assert inventory["test_set_access_count"] == 0
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    assert not any("datasets/" in name or "predictions" in name for name in names)
