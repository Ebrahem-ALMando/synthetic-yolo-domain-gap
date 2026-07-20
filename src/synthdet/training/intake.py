"""Strict intake validation for returned Sprint 4B training results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from synthdet.synthetic.contracts import sha256_file, stable_json_hash
from synthdet.training.experiments import REGIME_COUNTS

ARCHIVE_NAME = "sprint4b_training_results.zip"
CLASS_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
VALIDATION_NOTICE = "NON-FINAL — VALIDATION SET ONLY"
REQUIRED_EXTERNAL_FILES = (
    ARCHIVE_NAME,
    f"{ARCHIVE_NAME}.sha256",
    f"{ARCHIVE_NAME}.inventory.json",
    "training_completion_manifest.json",
    "final_profile.json",
)
REQUIRED_RUN_FILES = (
    "run_metadata.json",
    "resolved_training_config.yaml",
    "ultralytics/results.csv",
    "ultralytics/results.png",
    "ultralytics/weights/best.pt",
    "ultralytics/weights/last.pt",
)
SECRET_NAMES = {".env", "credentials.json", "service_account.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON/YAML mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _load_yaml(path: Path) -> dict[str, Any]:
    return _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _safe_member_path(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if relative.parts[0].endswith(":") or any(part in {"", "."} for part in relative.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return relative


def _stream_digest(handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _validate_inventory_entry(value: object) -> dict[str, Any]:
    entry = _require_mapping(value, "archive inventory entry")
    if set(entry) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"Malformed archive inventory entry: {entry.get('path', '<unknown>')}")
    path = entry["path"]
    digest = entry["sha256"]
    size = entry["size_bytes"]
    if not isinstance(path, str):
        raise ValueError("Archive inventory path must be a string")
    _safe_member_path(path)
    if not isinstance(digest, str) or not HASH_PATTERN.fullmatch(digest):
        raise ValueError(f"Invalid inventory SHA-256: {path}")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Invalid inventory size: {path}")
    return entry


def validate_return_archive(artifact_dir: Path) -> dict[str, Any]:
    """Validate the external sidecars and every ZIP member before extraction."""

    missing = [name for name in REQUIRED_EXTERNAL_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing Sprint 4B return files: " + ", ".join(missing))
    empty = [name for name in REQUIRED_EXTERNAL_FILES if (artifact_dir / name).stat().st_size <= 0]
    if empty:
        raise ValueError("Empty Sprint 4B return files: " + ", ".join(empty))

    archive_path = artifact_dir / ARCHIVE_NAME
    checksum_tokens = (artifact_dir / f"{ARCHIVE_NAME}.sha256").read_text(
        encoding="utf-8"
    ).split()
    if checksum_tokens != [checksum_tokens[0], ARCHIVE_NAME] or not HASH_PATTERN.fullmatch(
        checksum_tokens[0]
    ):
        raise ValueError("Malformed archive SHA-256 sidecar")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != checksum_tokens[0]:
        raise ValueError("Returned archive SHA-256 does not match its sidecar")

    external = _load_json(artifact_dir / f"{ARCHIVE_NAME}.inventory.json")
    required = {
        "archive_sha256",
        "contains_dataset_images",
        "contains_secrets",
        "contains_test_outputs",
        "inventory",
        "test_set_access_count",
        "training_identity",
    }
    missing_keys = sorted(required - set(external))
    if missing_keys:
        raise ValueError("External inventory is missing: " + ", ".join(missing_keys))
    if external["archive_sha256"] != archive_sha256:
        raise ValueError("External inventory archive hash mismatch")
    if any(
        external[key] is not False
        for key in ("contains_dataset_images", "contains_secrets", "contains_test_outputs")
    ):
        raise ValueError("External inventory declares prohibited content")
    if external["test_set_access_count"] != 0:
        raise ValueError("Returned training archive does not prove zero test access")
    if not isinstance(external["training_identity"], str) or not HASH_PATTERN.fullmatch(
        external["training_identity"]
    ):
        raise ValueError("External inventory training identity is invalid")
    raw_entries = external["inventory"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("External inventory must contain files")
    entries = [_validate_inventory_entry(value) for value in raw_entries]
    by_path = {entry["path"]: entry for entry in entries}
    if len(by_path) != len(entries):
        raise ValueError("External inventory contains duplicate paths")

    allowed_plot_names = {
        "boxf1_curve.png",
        "boxpr_curve.png",
        "boxp_curve.png",
        "boxr_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "non_final_validation_results.png",
    }
    for entry in entries:
        relative = _safe_member_path(entry["path"])
        lowered = [part.lower() for part in relative.parts]
        if "datasets" in lowered or "test" in lowered or any(
            part.startswith(("test_predictions", "test-results")) for part in lowered
        ):
            raise ValueError(f"Dataset/test content is prohibited: {entry['path']}")
        suffix = Path(relative.name).suffix.lower()
        if relative.name.lower() in SECRET_NAMES or suffix in SECRET_SUFFIXES:
            raise ValueError(f"Secret-like archive file is prohibited: {entry['path']}")
        if suffix in {".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"}:
            raise ValueError(f"Raw image is prohibited from the return archive: {entry['path']}")
        if suffix == ".png" and relative.name.lower() not in allowed_plot_names:
            raise ValueError(f"Unexpected PNG may contain raw data: {entry['path']}")

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        member_names = [member.filename for member in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("ZIP contains duplicate member paths")
        expected_names = set(by_path) | {"results_archive_inventory.json"}
        if set(member_names) != expected_names:
            missing_members = sorted(expected_names - set(member_names))
            extras = sorted(set(member_names) - expected_names)
            raise ValueError(
                f"ZIP member/inventory mismatch; missing={missing_members}, extras={extras}"
            )
        for member in members:
            _safe_member_path(member.filename)
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"ZIP symlink is prohibited: {member.filename}")
            if member.is_dir():
                raise ValueError(f"Unexpected directory entry in ZIP: {member.filename}")
            if member.filename == "results_archive_inventory.json":
                continue
            entry = by_path[member.filename]
            if member.file_size != entry["size_bytes"]:
                raise ValueError(f"ZIP member size mismatch: {member.filename}")
            with archive.open(member) as handle:
                if _stream_digest(handle) != entry["sha256"]:
                    raise ValueError(f"ZIP member SHA-256 mismatch: {member.filename}")
            if member.file_size <= 1024 * 1024 and Path(member.filename).suffix.lower() in {
                ".json",
                ".yaml",
                ".yml",
                ".csv",
                ".txt",
            }:
                payload = archive.read(member)
                if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
                    raise ValueError(f"Secret pattern found in archive: {member.filename}")
        internal = _require_mapping(
            json.loads(archive.read("results_archive_inventory.json")),
            "internal results inventory",
        )
    for key in required - {"archive_sha256"}:
        if internal.get(key) != external.get(key):
            raise ValueError(f"Internal/external inventory mismatch: {key}")

    completion_entry = by_path.get("completion/training_completion_manifest.json")
    if completion_entry is None:
        raise ValueError("Completion manifest is absent from the archive inventory")
    external_completion = artifact_dir / "training_completion_manifest.json"
    if (
        external_completion.stat().st_size != completion_entry["size_bytes"]
        or sha256_file(external_completion) != completion_entry["sha256"]
    ):
        raise ValueError("External completion manifest differs from archived manifest")

    return {
        "archive_name": ARCHIVE_NAME,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "inventory_file_count": len(entries),
        "inventory_total_bytes": sum(entry["size_bytes"] for entry in entries),
        "training_identity": external["training_identity"],
        "entries": entries,
    }


def extract_validated_archive(archive_path: Path, destination: Path) -> None:
    """Extract a previously validated ZIP atomically without overwriting prior intake."""

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sprint4b-extracting-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = _safe_member_path(member.filename)
                target = (staging / Path(*relative.parts)).resolve()
                if staging.resolve() not in target.parents:
                    raise ValueError(f"Archive member escapes extraction root: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _numeric_results(path: Path) -> tuple[list[dict[str, float]], list[str], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        raw_rows = list(reader)
    if not headers or not raw_rows:
        raise ValueError(f"Training result is empty or malformed: {path}")
    rows: list[dict[str, float]] = []
    for row_number, raw in enumerate(raw_rows, start=1):
        parsed: dict[str, float] = {}
        for key in headers:
            try:
                value = float(raw[key].strip())
            except (AttributeError, KeyError, ValueError) as error:
                raise ValueError(f"Non-numeric result at row {row_number}, column {key}") from error
            if not math.isfinite(value):
                raise ValueError(f"NaN/Inf result at row {row_number}, column {key}")
            parsed[key] = value
        rows.append(parsed)
    metric = next(
        (key for key in headers if "map50-95" in key.lower().replace(" ", "")),
        "",
    )
    if not metric:
        raise ValueError(f"mAP@50-95 validation column is absent: {path}")
    epoch_key = next((key for key in headers if key.strip().lower() == "epoch"), "")
    if not epoch_key:
        raise ValueError(f"Epoch column is absent: {path}")
    epochs = [row[epoch_key] for row in rows]
    if len(set(epochs)) != len(epochs) or any(
        right - left != 1 for left, right in zip(epochs, epochs[1:], strict=False)
    ):
        raise ValueError(f"Epoch sequence is missing or duplicated: {path}")
    return rows, headers, metric


def _metric(row: dict[str, float], fragment: str) -> float:
    key = next(
        (name for name in row if fragment in name.lower().replace(" ", "")),
        "",
    )
    if not key:
        raise ValueError(f"Validation metric is missing: {fragment}")
    return row[key]


def _checkpoint_facts(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    loaded = YOLO(str(path), task="detect")
    names_value = loaded.names
    names = (
        [str(names_value[index]) for index in sorted(names_value)]
        if isinstance(names_value, dict)
        else [str(value) for value in names_value]
    )
    model = loaded.model
    yaml_config = getattr(model, "yaml", {})
    if getattr(loaded, "task", None) != "detect" or type(model).__name__ != "DetectionModel":
        raise ValueError(f"Checkpoint is not an Ultralytics detection model: {path}")
    if names != CLASS_NAMES or int(yaml_config.get("nc", -1)) != len(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order/count mismatch: {path}")
    if yaml_config.get("scale") != "n":
        raise ValueError(f"Checkpoint is not the YOLO11 nano scale: {path}")
    return {
        "class_names": names,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_type": type(model).__name__,
        "scale": yaml_config.get("scale"),
        "task": loaded.task,
    }


def _assert_equal(values: Iterable[object], label: str) -> object:
    values_list = list(values)
    if not values_list or any(value != values_list[0] for value in values_list[1:]):
        raise ValueError(f"Cross-regime mismatch: {label}")
    return values_list[0]


def validate_extracted_runs(
    repository_root: Path,
    artifact_dir: Path,
    extracted: Path,
    archive_validation: dict[str, Any],
    *,
    load_checkpoints: bool = True,
) -> dict[str, Any]:
    """Validate scientific identity, configuration, CSVs, hashes, and checkpoints."""

    external_manifest = _load_json(artifact_dir / "training_completion_manifest.json")
    archived_manifest = _load_json(extracted / "completion/training_completion_manifest.json")
    if external_manifest != archived_manifest:
        raise ValueError("External and extracted completion manifests differ")
    profile = _load_json(artifact_dir / "final_profile.json")
    if external_manifest.get("hardware_profile") != profile:
        raise ValueError("Final profile differs from the completion manifest profile")
    if external_manifest.get("status") != "completed" or external_manifest.get(
        "test_set_access_count"
    ) != 0:
        raise ValueError("Sprint 4B completion/zero-test-access status is invalid")
    if external_manifest.get("combined_sprint4b_training_identity") != archive_validation.get(
        "training_identity"
    ):
        raise ValueError("Training identity differs between completion and archive inventory")

    common = _load_yaml(repository_root / "configs/training/common.yaml")
    if profile.get("status") != "frozen" or profile.get("profile_name") not in {
        "standard",
        "low_memory",
    }:
        raise ValueError("Returned GPU profile is not one recognized frozen profile")
    declared_profile = common["hardware_profiles"][profile["profile_name"]]
    if int(profile.get("batch", -1)) != int(declared_profile["batch"]):
        raise ValueError("Returned batch differs from the frozen profile")
    if int(profile.get("imgsz", -1)) != 640 or profile.get("identities") != common["identities"]:
        raise ValueError("Returned profile size/identities differ from the frozen protocol")
    profile_inputs = {
        key: profile[key]
        for key in (
            "profile_name",
            "batch",
            "imgsz",
            "device",
            "base_weight_sha256",
            "identities",
            "preflight",
        )
    }
    if stable_json_hash(profile_inputs) != profile.get("profile_identity"):
        raise ValueError("Returned hardware profile identity is invalid")
    if profile.get("test_set_access_count") != 0 or any(
        event.get("test_set_access_count") != 0 for event in profile.get("preflight", [])
    ):
        raise ValueError("GPU profile does not prove zero test access")
    environment = _require_mapping(profile.get("environment"), "hardware environment")
    for key in ("gpu_model", "nvidia_smi", "software_versions"):
        if not environment.get(key):
            raise ValueError(f"GPU environment metadata is missing: {key}")

    run_records = external_manifest.get("runs")
    if not isinstance(run_records, list):
        raise ValueError("Completion manifest runs must be a list")
    by_regime = {record.get("regime"): record for record in run_records if isinstance(record, dict)}
    expected_regimes = list(REGIME_COUNTS)
    if len(by_regime) != len(run_records) or set(by_regime) != set(expected_regimes):
        raise ValueError("Completion manifest must contain exactly the five frozen regimes")

    inventory = {entry["path"]: entry for entry in archive_validation["entries"]}
    rows: list[dict[str, Any]] = []
    normalized_configs: list[dict[str, Any]] = []
    csv_headers: list[list[str]] = []
    parameter_counts: list[int] = []
    revisions: list[str] = []
    for regime in expected_regimes:
        record = by_regime[regime]
        run_root = extracted / "runs" / regime
        missing = [
            relative for relative in REQUIRED_RUN_FILES if not (run_root / relative).is_file()
        ]
        if missing:
            raise ValueError(f"{regime}: required run files are missing: {', '.join(missing)}")
        metadata = _load_json(run_root / "run_metadata.json")
        resolved = _load_yaml(run_root / "resolved_training_config.yaml")
        if (
            metadata.get("status") != "completed"
            or metadata.get("mode") != "final"
            or metadata.get("regime") != regime
            or metadata.get("test_set_used") is not False
            or metadata.get("test_set_access_count") != 0
        ):
            raise ValueError(f"{regime}: completion/mode/zero-test-access metadata is invalid")
        if metadata.get("identities") != common["identities"]:
            raise ValueError(f"{regime}: frozen scientific identities differ")
        revision = metadata.get("git_revision")
        if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
            raise ValueError(f"{regime}: repository revision is invalid")
        revisions.append(revision)
        if record.get("git_revision") != revision:
            raise ValueError(f"{regime}: completion/run revision mismatch")
        if record.get("run_status") != "completed" or record.get("test_set_access_count") != 0:
            raise ValueError(f"{regime}: completion run status is invalid")
        if record.get("model_architecture") != "YOLO11n" or record.get("seed") != 42:
            raise ValueError(f"{regime}: architecture/seed mismatch")
        if record.get("hardware_profile_identity") != profile["profile_identity"]:
            raise ValueError(f"{regime}: shared hardware profile identity mismatch")
        if record.get("base_weight_hash") != profile["base_weight_sha256"]:
            raise ValueError(f"{regime}: base-weight identity mismatch")
        if metadata.get("hardware_profile", {}).get("profile_identity") != profile[
            "profile_identity"
        ]:
            raise ValueError(f"{regime}: run metadata profile mismatch")
        if metadata.get("model", {}).get("pretrained_weight_sha256") != profile[
            "base_weight_sha256"
        ]:
            raise ValueError(f"{regime}: run metadata base-weight mismatch")
        if int(resolved.get("batch", -1)) != int(profile["batch"]):
            raise ValueError(f"{regime}: batch differs from shared profile")
        expected_config = common["training"]
        for key in (
            "epochs",
            "imgsz",
            "seed",
            "deterministic",
            "optimizer",
            "patience",
            "workers",
            "conf",
            "iou",
        ):
            if resolved.get(key) != expected_config[key]:
                raise ValueError(f"{regime}: resolved training setting differs: {key}")
        for key, value in common["augmentation"].items():
            if resolved.get(key) != value:
                raise ValueError(f"{regime}: resolved augmentation setting differs: {key}")
        if resolved.get("resume") is not False or resolved.get("batch") == -1:
            raise ValueError(f"{regime}: resume/auto-batch policy is invalid")
        normalized_configs.append(
            {
                key: value
                for key, value in resolved.items()
                if key not in {"data", "name", "project"}
            }
        )

        results_path = run_root / "ultralytics/results.csv"
        result_rows, headers, metric = _numeric_results(results_path)
        csv_headers.append(headers)
        if len(result_rows) != int(record.get("completed_epochs", -1)) or len(result_rows) != 50:
            raise ValueError(f"{regime}: result epoch count is not the completed 50 epochs")
        best_index = max(range(len(result_rows)), key=lambda index: result_rows[index][metric])
        best_row = result_rows[best_index]
        if int(record.get("best_epoch", -1)) != best_index + 1:
            raise ValueError(f"{regime}: best validation epoch is inconsistent")
        metric_checks = {
            "validation_precision": _metric(best_row, "precision"),
            "validation_recall": _metric(best_row, "recall"),
            "validation_map50": _metric(best_row, "map50(b)"),
            "validation_map50_95": best_row[metric],
        }
        for key, value in metric_checks.items():
            if not math.isclose(float(record[key]), value, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{regime}: completion validation metric differs: {key}")

        checkpoint_facts: dict[str, Any] = {
            "class_names": CLASS_NAMES,
            "parameter_count": None,
            "model_type": "not_loaded",
            "scale": None,
            "task": None,
        }
        checkpoint_records = (
            ("best.pt", "best_pt_sha256"),
            ("last.pt", "last_pt_sha256"),
        )
        for checkpoint_name, record_key in checkpoint_records:
            checkpoint = run_root / "ultralytics/weights" / checkpoint_name
            archive_path = f"runs/{regime}/ultralytics/weights/{checkpoint_name}"
            entry = inventory[archive_path]
            digest = sha256_file(checkpoint)
            if checkpoint.stat().st_size != entry["size_bytes"] or digest != entry["sha256"]:
                raise ValueError(f"{regime}: extracted checkpoint differs from archive inventory")
            if digest != record.get(record_key):
                raise ValueError(f"{regime}: checkpoint differs from completion manifest")
            if load_checkpoints:
                loaded_facts = _checkpoint_facts(checkpoint)
                if checkpoint_name == "best.pt":
                    checkpoint_facts = loaded_facts
                elif loaded_facts["parameter_count"] != checkpoint_facts["parameter_count"]:
                    raise ValueError(f"{regime}: best/last architectures differ")
        if checkpoint_facts["parameter_count"] is not None:
            parameter_counts.append(checkpoint_facts["parameter_count"])

        if sha256_file(results_path) != record.get("results_csv_sha256"):
            raise ValueError(f"{regime}: results CSV hash differs from completion manifest")
        rows.append(
            {
                "notice": VALIDATION_NOTICE,
                "regime": regime,
                "repository_revision": revision,
                "profile_name": profile["profile_name"],
                "batch": profile["batch"],
                "epochs": len(result_rows),
                "best_epoch": best_index + 1,
                "duration_seconds": float(record["duration_seconds"]),
                "validation_precision": metric_checks["validation_precision"],
                "validation_recall": metric_checks["validation_recall"],
                "validation_map50": metric_checks["validation_map50"],
                "validation_map50_95": metric_checks["validation_map50_95"],
                "best_pt_path": (
                    "artifacts/external_training/sprint4b-v2/extracted/runs/"
                    f"{regime}/ultralytics/weights/best.pt"
                ),
                "best_pt_sha256": record["best_pt_sha256"],
                "best_pt_size_bytes": (
                    run_root / "ultralytics/weights/best.pt"
                ).stat().st_size,
                "last_pt_sha256": record["last_pt_sha256"],
                "results_csv_sha256": record["results_csv_sha256"],
                "parameter_count": checkpoint_facts["parameter_count"],
                "model_type": checkpoint_facts["model_type"],
                "model_scale": checkpoint_facts["scale"],
                "class_names": checkpoint_facts["class_names"],
            }
        )

    _assert_equal(revisions, "repository revision")
    _assert_equal(normalized_configs, "resolved training configuration")
    _assert_equal(csv_headers, "results CSV headers")
    if load_checkpoints:
        _assert_equal(parameter_counts, "checkpoint parameter count")

    identity_inputs = {
        "profile_identity": profile["profile_identity"],
        "experiment_design_identity": common["identities"]["experiment_design"],
        "runs": [
            {
                key: by_regime[regime][key]
                for key in (
                    "regime",
                    "regime_manifest_hash",
                    "best_pt_sha256",
                    "last_pt_sha256",
                    "results_csv_sha256",
                )
            }
            for regime in expected_regimes
        ],
    }
    training_identity = stable_json_hash(identity_inputs)
    if training_identity != external_manifest["combined_sprint4b_training_identity"]:
        raise ValueError("Combined Sprint 4B training identity is invalid")
    return {
        "status": "verified",
        "notice": VALIDATION_NOTICE,
        "training_identity": training_identity,
        "repository_revision": revisions[0],
        "profile": profile,
        "configuration_equality": True,
        "checkpoint_loadability_verified": load_checkpoints,
        "class_names": CLASS_NAMES,
        "runs": rows,
    }


def write_intake_reports(
    report_dir: Path, archive_validation: dict[str, Any], validation: dict[str, Any]
) -> list[Path]:
    """Write tracked validation-only reports without copying weights or external archives."""

    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    runs = validation["runs"]
    summary_path = report_dir / "sprint4b_v2_validation_summary.csv"
    summary_fields = (
        "notice",
        "regime",
        "validation_precision",
        "validation_recall",
        "validation_map50",
        "validation_map50_95",
        "best_epoch",
        "epochs",
        "duration_seconds",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in summary_fields})

    checkpoint_path = report_dir / "sprint4b_v2_checkpoint_inventory.csv"
    checkpoint_fields = (
        "regime",
        "best_pt_path",
        "best_pt_sha256",
        "best_pt_size_bytes",
        "last_pt_sha256",
        "results_csv_sha256",
        "parameter_count",
        "model_type",
        "model_scale",
        "class_order",
    )
    with checkpoint_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=checkpoint_fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    **{key: run[key] for key in checkpoint_fields if key != "class_order"},
                    "class_order": "|".join(run["class_names"]),
                }
            )

    figure_path = report_dir / "sprint4b_v2_validation_summary.png"
    names = [run["regime"] for run in runs]
    map_values = [run["validation_map50_95"] for run in runs]
    map50_values = [run["validation_map50"] for run in runs]
    positions = list(range(len(runs)))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar([value - 0.18 for value in positions], map_values, width=0.36, label="mAP@50-95")
    axis.bar([value + 0.18 for value in positions], map50_values, width=0.36, label="mAP@50")
    axis.set_xticks(positions, names, rotation=18)
    axis.set_ylim(0, max(map50_values) * 1.2)
    axis.set_ylabel("Validation metric")
    axis.set_title(VALIDATION_NOTICE)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    environment_path = report_dir / "sprint4b_v2_environment_summary.json"
    environment_path.write_text(
        json.dumps(
            {
                "notice": VALIDATION_NOTICE,
                "repository_revision": validation["repository_revision"],
                "training_identity": validation["training_identity"],
                "hardware_profile": validation["profile"],
                "class_names": CLASS_NAMES,
                "checkpoint_loadability_verified": validation[
                    "checkpoint_loadability_verified"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = report_dir / "sprint4b_v2_intake_report.md"
    table_rows = "\n".join(
        "| {regime} | {validation_precision:.5f} | {validation_recall:.5f} | "
        "{validation_map50:.5f} | {validation_map50_95:.5f} | {best_epoch} | "
        "{duration_seconds:.3f} |".format(**run)
        for run in runs
    )
    report_path.write_text(
        "# Sprint 4B V2 Artifact Intake\n\n"
        f"> **{VALIDATION_NOTICE}.** These values come from the shared 140-image real validation "
        "split. They are not protected-test results and do not select the final model.\n\n"
        "## Intake verdict\n\n"
        "The returned CUDA bundle passed checksum, safe-path, inventory, per-file size/hash, "
        "forbidden-content, five-regime identity, configuration equality, CSV integrity, and "
        "checkpoint loadability checks. No raw dataset image, secret, final-test output, unsafe "
        "symlink, duplicate path, or non-zero test access was found.\n\n"
        f"- Archive: `{archive_validation['archive_name']}` "
        f"({archive_validation['archive_size_bytes']} bytes)\n"
        f"- Archive SHA-256: `{archive_validation['archive_sha256']}`\n"
        f"- Inventoried files: {archive_validation['inventory_file_count']}\n"
        f"- Training identity: `{validation['training_identity']}`\n"
        f"- Training source revision: `{validation['repository_revision']}`\n"
        f"- Frozen profile: `{validation['profile']['profile_name']}`; batch "
        f"{validation['profile']['batch']}; image size {validation['profile']['imgsz']}\n"
        f"- GPU: `{validation['profile']['environment']['gpu_model']}`\n"
        "- Completed regimes: 5/5, 50 epochs each\n"
        "- Training-time protected-test access count: 0\n"
        f"- Class order: `{', '.join(CLASS_NAMES)}`\n\n"
        "## Validation-only summary\n\n"
        "| Regime | Precision | Recall | mAP@50 | mAP@50-95 | Best epoch | Duration (s) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        "## Scientific interpretation boundary\n\n"
        "The validation comparison is diagnostic evidence that training completed and that all "
        "outputs are parseable. It is not a final ranking. The final recommendation remains locked "
        "until the Sprint 5 contract is committed and one complete five-model campaign is run on "
        "the fixed protected real test split.\n",
        encoding="utf-8",
    )

    generated = [summary_path, checkpoint_path, figure_path, environment_path, report_path]
    hash_path = report_dir / "sprint4b_v2_hash_report.json"
    hash_path.write_text(
        json.dumps(
            {
                "notice": VALIDATION_NOTICE,
                "archive": {
                    key: archive_validation[key]
                    for key in (
                        "archive_name",
                        "archive_size_bytes",
                        "archive_sha256",
                        "inventory_file_count",
                        "inventory_total_bytes",
                    )
                },
                "training_identity": validation["training_identity"],
                "archive_inventory": archive_validation["entries"],
                "generated_report_hashes": {
                    path.name: sha256_file(path) for path in generated
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*generated, hash_path]
