"""Validation, completion manifests, and safe exports for full primary runs."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import yaml

from synthdet.synthetic.contracts import sha256_file, stable_json_hash
from synthdet.training.experiments import REGIME_COUNTS

SECRET_NAMES = {".env", "credentials.json", "service_account.json"}

REQUIRED_RUN_FILES = (
    "run_metadata.json",
    "resolved_training_config.yaml",
    "ultralytics/results.csv",
    "ultralytics/results.png",
    "ultralytics/weights/best.pt",
    "ultralytics/weights/last.pt",
)
VALIDATION_NOTICE = "NON-FINAL VALIDATION RESULTS"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def load_frozen_profile(path: Path, common: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status",
        "profile_name",
        "batch",
        "imgsz",
        "device",
        "environment",
        "base_weight_sha256",
        "identities",
        "profile_identity",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError("Frozen hardware profile is missing: " + ", ".join(missing))
    if profile["status"] != "frozen":
        raise ValueError("Hardware profile is not frozen")
    expected_profiles = {
        "standard": common["hardware_profiles"]["standard"],
        "low_memory": common["hardware_profiles"]["low_memory"],
    }
    selected = expected_profiles.get(profile["profile_name"])
    if selected is None:
        raise ValueError("Unknown frozen hardware profile")
    if int(profile["batch"]) != int(selected["batch"]) or int(profile["imgsz"]) != int(
        selected["imgsz"]
    ):
        raise ValueError("Frozen hardware profile does not match the declared profile")
    if profile["identities"] != common["identities"]:
        raise ValueError("Hardware profile has the wrong frozen identities")
    identity_inputs = {
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
    if stable_json_hash(identity_inputs) != profile["profile_identity"]:
        raise ValueError("Hardware-profile identity mismatch")
    return profile


def _read_results(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training results contain no epochs: {path}")
    for row_number, row in enumerate(rows, start=1):
        for key, value in row.items():
            try:
                number = float(value.strip())
            except (AttributeError, ValueError) as error:
                raise ValueError(
                    f"Non-numeric result at epoch {row_number}, column {key}"
                ) from error
            if not math.isfinite(number):
                raise ValueError(f"NaN/Inf result at epoch {row_number}, column {key}")
    metric = next(
        (key for key in rows[0] if "map50-95" in key.lower().replace(" ", "")),
        "",
    )
    if not metric:
        raise ValueError("results.csv lacks a mAP50-95 validation column")
    return rows, metric


def validate_completed_run(
    run_dir: Path,
    regime: str,
    common: dict[str, Any],
    profile: dict[str, Any],
    expected_manifest_hash: str,
) -> dict[str, Any]:
    for path in run_dir.rglob("*"):
        lowered = [part.lower() for part in path.relative_to(run_dir).parts]
        if any(
            part == "test" or part.startswith("test_predictions") or part.startswith("test-results")
            for part in lowered
        ):
            raise ValueError(f"{regime}: prohibited test output exists: {path.name}")
    missing = [relative for relative in REQUIRED_RUN_FILES if not (run_dir / relative).is_file()]
    if missing:
        raise ValueError(f"{regime}: incomplete run files: {', '.join(missing)}")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"{regime}: run status is not completed")
    if metadata.get("regime") != regime or metadata.get("mode") != "final":
        raise ValueError(f"{regime}: run mode/regime metadata mismatch")
    if (
        metadata.get("test_set_used") is not False
        or int(metadata.get("test_set_access_count", -1)) != 0
    ):
        raise ValueError(f"{regime}: zero test-set access is not proven")
    if metadata.get("regime_manifest_hash") != expected_manifest_hash:
        raise ValueError(f"{regime}: manifest hash mismatch")
    if metadata.get("identities") != common["identities"]:
        raise ValueError(f"{regime}: frozen identities mismatch")
    resolved = load_yaml(run_dir / "resolved_training_config.yaml")
    if int(resolved["batch"]) != int(profile["batch"]) or int(resolved["imgsz"]) != 640:
        raise ValueError(f"{regime}: hardware profile differs from frozen selection")
    if resolved.get("resume") is not False or resolved.get("optimizer") != "AdamW":
        raise ValueError(f"{regime}: resolved optimizer/resume policy mismatch")
    if metadata.get("hardware_profile", {}).get("profile_identity") != profile["profile_identity"]:
        raise ValueError(f"{regime}: hardware-profile identity was not recorded")
    if metadata.get("model", {}).get("pretrained_weight_sha256") != profile["base_weight_sha256"]:
        raise ValueError(f"{regime}: base-weight hash mismatch")
    results_path = run_dir / "ultralytics/results.csv"
    rows, metric = _read_results(results_path)
    requested_epochs = int(resolved["epochs"])
    completed_epochs = len(rows)
    if completed_epochs > requested_epochs:
        raise ValueError(f"{regime}: completed more epochs than requested")
    best_index = max(range(completed_epochs), key=lambda index: float(rows[index][metric].strip()))
    best_row = rows[best_index]

    def metric_value(fragment: str) -> float | None:
        key = next((name for name in best_row if fragment in name.lower().replace(" ", "")), None)
        return float(best_row[key].strip()) if key else None

    best_path = run_dir / "ultralytics/weights/best.pt"
    last_path = run_dir / "ultralytics/weights/last.pt"
    return {
        "regime": regime,
        "run_directory": run_dir.as_posix(),
        "run_status": "completed",
        "git_revision": metadata["git_revision"],
        "dataset_identity": common["identities"]["real_split"],
        "synthetic_identity": common["identities"]["synthetic_pool"],
        "experiment_design_identity": common["identities"]["experiment_design"],
        "regime_manifest_hash": expected_manifest_hash,
        "model_architecture": common["model"]["architecture"],
        "base_weight_hash": profile["base_weight_sha256"],
        "hardware_profile": profile["profile_name"],
        "hardware_profile_identity": profile["profile_identity"],
        "gpu_model": profile["environment"]["gpu_model"],
        "software_versions": profile["environment"]["software_versions"],
        "seed": int(resolved["seed"]),
        "requested_epochs": requested_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_index + 1,
        "early_stopped": completed_epochs < requested_epochs,
        "best_pt_path": best_path.as_posix(),
        "best_pt_sha256": sha256_file(best_path),
        "last_pt_path": last_path.as_posix(),
        "last_pt_sha256": sha256_file(last_path),
        "results_csv_path": results_path.as_posix(),
        "results_csv_sha256": sha256_file(results_path),
        "started_at_utc": metadata["started_at_utc"],
        "ended_at_utc": metadata["ended_at_utc"],
        "duration_seconds": metadata["duration_seconds"],
        "test_set_access_count": 0,
        "validation_precision": metric_value("precision"),
        "validation_recall": metric_value("recall"),
        "validation_map50": metric_value("map50(b)"),
        "validation_map50_95": float(best_row[metric].strip()),
    }


def build_completion_manifest(
    runs_root: Path,
    profile_path: Path,
    common_path: Path,
    experiment_metadata_path: Path,
) -> dict[str, Any]:
    common = load_yaml(common_path)
    profile = load_frozen_profile(profile_path, common)
    experiment = json.loads(experiment_metadata_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for regime in REGIME_COUNTS:
        candidates = sorted(runs_root.glob(f"{regime}-*"))
        valid: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                valid.append(
                    validate_completed_run(
                        candidate,
                        regime,
                        common,
                        profile,
                        experiment["regime_manifest_hashes"][f"{regime}.csv"],
                    )
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
        if not valid:
            raise ValueError(f"No valid completed final run exists for {regime}")
        records.append(max(valid, key=lambda row: row["ended_at_utc"]))
    identity_inputs = {
        "profile_identity": profile["profile_identity"],
        "experiment_design_identity": common["identities"]["experiment_design"],
        "runs": [
            {
                key: row[key]
                for key in (
                    "regime",
                    "regime_manifest_hash",
                    "best_pt_sha256",
                    "last_pt_sha256",
                    "results_csv_sha256",
                )
            }
            for row in records
        ],
    }
    return {
        "status": "completed",
        "test_set_access_count": 0,
        "validation_results_notice": VALIDATION_NOTICE,
        "hardware_profile": profile,
        "runs": records,
        "combined_sprint4b_training_identity": stable_json_hash(identity_inputs),
    }


def write_validation_table(manifest: dict[str, Any], path: Path) -> None:
    fields = (
        "notice",
        "regime",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "best_epoch",
        "duration_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in manifest["runs"]:
            writer.writerow(
                {
                    "notice": VALIDATION_NOTICE,
                    "regime": row["regime"],
                    "precision": row["validation_precision"],
                    "recall": row["validation_recall"],
                    "mAP50": row["validation_map50"],
                    "mAP50_95": row["validation_map50_95"],
                    "best_epoch": row["best_epoch"],
                    "duration_seconds": row["duration_seconds"],
                }
            )


def write_validation_figure(manifest: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["regime"] for row in manifest["runs"]]
    values = [row["validation_map50_95"] for row in manifest["runs"]]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, values, color="#2878B5")
    axis.set_ylabel("Validation mAP@50-95")
    axis.set_title(VALIDATION_NOTICE)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_results_archive(
    manifest: dict[str, Any],
    output: Path,
    completion_path: Path,
    validation_table: Path,
    validation_figure: Path | None = None,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite results archive: {output}")
    included: dict[str, Path] = {
        "completion/training_completion_manifest.json": completion_path,
        "completion/NON_FINAL_VALIDATION_RESULTS.csv": validation_table,
    }
    if validation_figure is not None:
        included["completion/NON_FINAL_VALIDATION_RESULTS.png"] = validation_figure
    excluded_prefixes = ("train_batch", "val_batch", "labels", "predictions")
    for record in manifest["runs"]:
        run_dir = Path(record["run_directory"])
        for path in run_dir.rglob("*"):
            if not path.is_file() or path.name.lower().startswith(excluded_prefixes):
                continue
            lowered_parts = {part.lower() for part in path.parts}
            if "datasets" in lowered_parts or path.name.lower() in SECRET_NAMES:
                continue
            relative = f"runs/{record['regime']}/{path.relative_to(run_dir).as_posix()}"
            included[relative] = path
    inventory = [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(included.items())
    ]
    export_metadata = {
        "training_identity": manifest["combined_sprint4b_training_identity"],
        "test_set_access_count": 0,
        "contains_dataset_images": False,
        "contains_test_outputs": False,
        "contains_secrets": False,
        "inventory": inventory,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in included.items():
            archive.write(path, name)
        archive.writestr(
            "results_archive_inventory.json",
            json.dumps(export_metadata, indent=2, sort_keys=True) + "\n",
        )
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return {**export_metadata, "archive_sha256": digest}
