"""Safe, provenance-recording Ultralytics training runner."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import psutil
import yaml

from synthdet.config.loader import load_config
from synthdet.synthetic.contracts import (
    read_csv,
    sha256_file,
    stable_json_hash,
    verify_active_split,
)
from synthdet.training.environment import collect_environment
from synthdet.training.materialize import validate_yolo_label


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def validate_synthetic_dataset_descriptor(path: Path, class_names: list[str]) -> None:
    data = load_yaml(path)
    names = data.get("names", {})
    normalized_names = (
        [names[index] for index in sorted(names)] if isinstance(names, dict) else names
    )
    if normalized_names != class_names:
        raise ValueError("Synthetic source descriptor class order mismatch")
    if data.get("train") != "images" or data.get("val") is not None:
        raise ValueError("Synthetic source descriptor must remain explicitly train-only")
    if data.get("path") != "." or data.get("synthetic_train_only") is not True:
        raise ValueError("Synthetic source descriptor portability/role contract mismatch")


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    inventory = root / "training_bundle_inventory.json"
    if inventory.is_file():
        from synthdet.training.bundle import validate_extracted_bundle
        from synthdet.training.colab import resolve_expected_revision

        validate_extracted_bundle(root)
        return resolve_expected_revision(root)
    raise RuntimeError("Cannot determine source revision from Git or bundle inventory")


def verify_frozen_project_inputs(
    common: dict[str, Any], experiment_metadata: dict[str, Any], project_root: Path
) -> None:
    """Hard-fail unless every real, synthetic, and experiment identity remains frozen."""

    project = load_config(project_root / "configs/project.yaml")
    split_dir = project_root / project.dataset.paths.train_manifest.parent
    verify_active_split(split_dir, common["identities"]["real_split"])
    identity_mapping = {
        "real_split": "real_split_identity",
        "synthetic_pool": "synthetic_pool_identity",
        "object_bank": "object_bank_identity",
        "generator_configuration": "generator_configuration_identity",
        "experiment_design": "combined_experiment_design_identity",
    }
    for common_key, metadata_key in identity_mapping.items():
        if common["identities"][common_key] != experiment_metadata[metadata_key]:
            raise ValueError(f"Frozen identity mismatch: {common_key}")
    if project.experiments.design_identity != common["identities"]["experiment_design"]:
        raise ValueError("Project configuration has the wrong experiment-design identity")
    generation_path = project_root / project.synthetic.manifests / "generation_metadata.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if generation["combined_synthetic_pool_identity"] != common["identities"]["synthetic_pool"]:
        raise ValueError("Synthetic-pool identity mismatch")
    if generation["object_bank_identity"] != common["identities"]["object_bank"]:
        raise ValueError("Object-bank identity mismatch")
    if generation["configuration_hash"] != common["identities"]["generator_configuration"]:
        raise ValueError("Generator-configuration identity mismatch")
    actual_synthetic_hashes: dict[str, str] = {}
    for name, expected in generation["output_manifest_hashes"].items():
        candidate = (
            project_root / project.synthetic.output / "data.yaml"
            if name == "materialized_data.yaml"
            else project_root / project.synthetic.manifests / name
        )
        actual = sha256_file(candidate)
        actual_synthetic_hashes[name] = actual
        if actual != expected:
            raise ValueError(f"Frozen synthetic manifest hash mismatch: {name}")
    synthetic_descriptor = project_root / project.synthetic.output / "data.yaml"
    validate_synthetic_dataset_descriptor(
        synthetic_descriptor, common["dataset"]["class_names"]
    )
    if stable_json_hash(actual_synthetic_hashes) != common["identities"]["synthetic_pool"]:
        raise ValueError("Synthetic manifest identity reproduction failed")


def validate_runner_inputs(
    regime_config: dict[str, Any],
    common: dict[str, Any],
    experiment_metadata: dict[str, Any],
    project_root: Path,
    mode: str,
    verify_project: bool = True,
) -> dict[str, Any]:
    if verify_project:
        verify_frozen_project_inputs(common, experiment_metadata, project_root)
    regime = regime_config["regime"]
    manifest_path = project_root / regime_config["manifest"]
    if sha256_file(manifest_path) != regime_config["expected_manifest_hash"]:
        raise ValueError(f"Frozen regime manifest hash mismatch: {regime}")
    if (
        regime_config["expected_manifest_hash"]
        != experiment_metadata["regime_manifest_hashes"][manifest_path.name]
    ):
        raise ValueError(f"Regime hash disagrees with experiment metadata: {regime}")
    if (
        common["identities"]["experiment_design"]
        != experiment_metadata["combined_experiment_design_identity"]
    ):
        raise ValueError("Common configuration has the wrong experiment-design identity")
    view = project_root / regime_config["dataset_view"]
    if mode == "smoke":
        view = project_root / common["smoke"]["view_root"] / regime
    data_yaml = view / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset view is not materialized: {data_yaml}")
    data = load_yaml(data_yaml)
    if "test" in data or any("test" in str(data.get(key, "")).lower() for key in data):
        raise ValueError("Dataset configuration references a forbidden test path")
    names = data["names"]
    normalized = [names[index] for index in sorted(names)] if isinstance(names, dict) else names
    if normalized != common["dataset"]["class_names"]:
        raise ValueError("Dataset class order differs from the common training configuration")
    if "path" in data:
        raise ValueError("Regime data.yaml must resolve relative to its own directory")
    expected_yaml_paths = {"train": "train/images", "val": "val/images"}
    for key, expected_relative in expected_yaml_paths.items():
        value = data.get(key)
        if not isinstance(value, str) or value != expected_relative:
            raise ValueError(f"Regime data.yaml has the wrong {key} path")
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or "\\" in value:
            raise ValueError(f"Regime data.yaml contains a non-portable {key} path")
        if (view / value).resolve() != (view / expected_relative).resolve():
            raise ValueError(f"Regime data.yaml {key} path escapes its frozen view")
    expected_train = (
        int(common["smoke"]["train_subset_count"])
        if mode == "smoke"
        else int(common["dataset"]["train_budget"])
    )
    expected_validation = (
        int(common["smoke"]["validation_subset_count"])
        if mode == "smoke"
        else int(common["dataset"]["validation_count"])
    )
    train_images = sorted((view / "train/images").glob("*"))
    train_labels = sorted((view / "train/labels").glob("*.txt"))
    validation_images = sorted((view / "val/images").glob("*"))
    validation_labels = sorted((view / "val/labels").glob("*.txt"))
    if len(train_images) != expected_train or len(train_labels) != expected_train:
        raise ValueError(f"{regime} {mode} training view count mismatch")
    if (
        len(validation_images) != expected_validation
        or len(validation_labels) != expected_validation
    ):
        raise ValueError(f"{regime} {mode} validation view count mismatch")
    if {path.stem for path in train_images} != {path.stem for path in train_labels}:
        raise ValueError(f"{regime} training image-label pairing mismatch")
    if {path.stem for path in validation_images} != {path.stem for path in validation_labels}:
        raise ValueError(f"{regime} validation image-label pairing mismatch")
    annotation_errors = [
        error
        for label in train_labels + validation_labels
        for error in validate_yolo_label(label, len(normalized))
    ]
    if annotation_errors:
        raise ValueError("Invalid YOLO annotation: " + annotation_errors[0])
    project = load_config(project_root / "configs/project.yaml") if verify_project else None
    split_dir = project_root / project.dataset.paths.train_manifest.parent if project else None
    test_hashes = (
        {row["content_hash"] for row in read_csv(split_dir / "real_test.csv")}
        if split_dir
        else set()
    )
    validation_hashes = (
        {row["content_hash"] for row in read_csv(split_dir / "real_val.csv")}
        if split_dir
        else {sha256_file(path) for path in validation_images}
    )
    validation_label_hashes = (
        {
            sha256_file(project_root / row["label_path"])
            for row in read_csv(split_dir / "real_val.csv")
        }
        if split_dir
        else {sha256_file(path) for path in validation_labels}
    )
    observed_train_hashes = {sha256_file(path) for path in train_images}
    observed_validation_hashes = {sha256_file(path) for path in validation_images}
    observed_validation_label_hashes = {sha256_file(path) for path in validation_labels}
    manifest_rows = read_csv(manifest_path)
    allowed_train_hashes = {row["image_hash"] for row in manifest_rows}
    allowed_train_label_hashes = {row["label_hash"] for row in manifest_rows}
    observed_train_label_hashes = {sha256_file(path) for path in train_labels}
    if not observed_train_hashes <= allowed_train_hashes:
        raise ValueError("Training view contains content outside its frozen regime manifest")
    if not observed_train_label_hashes <= allowed_train_label_hashes:
        raise ValueError("Training view contains labels outside its frozen regime manifest")
    if observed_train_hashes & test_hashes or observed_validation_hashes & test_hashes:
        raise ValueError("A materialized view contains protected test content")
    if not observed_validation_hashes <= validation_hashes:
        raise ValueError("Validation view contains content outside active real validation")
    if not observed_validation_label_hashes <= validation_label_hashes:
        raise ValueError("Validation view contains labels outside active real validation")
    if mode == "final" and observed_validation_hashes != validation_hashes:
        raise ValueError("Final validation view identity is incomplete")
    return {"regime": regime, "data_yaml": data_yaml, "manifest_path": manifest_path}


def resolved_training_arguments(
    common: dict[str, Any],
    mode: str,
    device: str,
    run_dir: Path,
    hardware_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = dict(common["training"])
    settings.update(common["augmentation"])
    if mode == "smoke":
        settings.update(common["smoke"]["overrides"])
    elif hardware_profile is None:
        raise ValueError("Final training requires a frozen CUDA hardware profile")
    else:
        settings["batch"] = int(hardware_profile["batch"])
        settings["imgsz"] = int(hardware_profile["imgsz"])
        settings["workers"] = int(
            common["hardware_profiles"][hardware_profile["profile_name"]]["workers"]
        )
    settings.update(
        {
            "device": device,
            "project": str(run_dir),
            "name": "ultralytics",
            "exist_ok": False,
            "resume": False,
            "verbose": True,
        }
    )
    return settings


def create_run_directory(base: Path, regime: str, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    parent = base / mode
    for suffix in range(1000):
        qualifier = "" if suffix == 0 else f"-{suffix:03d}"
        run_dir = parent / f"{regime}-{timestamp}{qualifier}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir
    raise FileExistsError(f"Could not allocate a unique run directory under {parent}")


def run_training(
    regime_config_path: Path,
    common_config_path: Path,
    metadata_path: Path,
    project_root: Path,
    mode: str,
    device: str,
    command: list[str],
    profile_path: Path | None = None,
    dry_run: bool = False,
    verify_project: bool = True,
) -> dict[str, Any]:
    common = load_yaml(common_config_path)
    regime_config = load_yaml(regime_config_path)
    experiment_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validated = validate_runner_inputs(
        regime_config,
        common,
        experiment_metadata,
        project_root,
        mode,
        verify_project=verify_project,
    )
    hardware_profile = None
    if mode == "final":
        if profile_path is None:
            raise ValueError("Final training requires --profile with frozen preflight metadata")
        from synthdet.training.completion import load_frozen_profile

        hardware_profile = load_frozen_profile(profile_path, common)
        if str(hardware_profile["device"]) != str(device):
            raise ValueError("Requested device differs from the frozen hardware profile")
    arguments = resolved_training_arguments(common, mode, device, Path("pending"), hardware_profile)
    if dry_run:
        return {
            "status": "dry_run_validated",
            "regime": validated["regime"],
            "mode": mode,
            "data_yaml": validated["data_yaml"].relative_to(project_root).as_posix(),
            "manifest_hash": sha256_file(validated["manifest_path"]),
            "resolved_arguments": arguments,
        }
    run_base = project_root / common["outputs"]["run_root"]
    run_dir = create_run_directory(run_base, validated["regime"], mode)
    arguments = resolved_training_arguments(common, mode, device, run_dir, hardware_profile)
    weight_path = project_root / common["model"]["pretrained_weights"]
    start = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "scientific_result": False if mode == "smoke" else None,
        "regime": validated["regime"],
        "command": shlex.join(command),
        "started_at_utc": start.isoformat(),
        "ended_at_utc": None,
        "duration_seconds": None,
        "git_revision": _git_revision(project_root),
        "regime_manifest_hash": sha256_file(validated["manifest_path"]),
        "identities": common["identities"],
        "model": {
            "architecture": common["model"]["architecture"],
            "pretrained_weights": common["model"]["pretrained_weights"],
            "pretrained_weight_sha256": sha256_file(weight_path) if weight_path.is_file() else None,
            "configuration_identity": stable_json_hash(common["model"]),
        },
        "environment": collect_environment(project_root),
        "resolved_arguments": arguments,
        "python": sys.version,
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "test_set_used": False,
        "test_set_access_count": 0,
        "hardware_profile": hardware_profile,
    }
    metadata_path_out = run_dir / "run_metadata.json"
    resolved_path = run_dir / "resolved_training_config.yaml"
    resolved_path.write_text(yaml.safe_dump(arguments, sort_keys=False), encoding="utf-8")
    metadata_path_out.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    try:
        from ultralytics import YOLO

        model = YOLO(common["model"]["pretrained_weights"])
        if weight_path.is_file():
            metadata["model"]["pretrained_weight_sha256"] = sha256_file(weight_path)
            metadata_path_out.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        result = model.train(data=str(validated["data_yaml"]), **arguments)
        metadata["status"] = "completed"
        metadata["ultralytics_save_dir"] = str(result.save_dir)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error_type"] = type(error).__name__
        metadata["error_message"] = str(error)
        raise
    finally:
        memory = psutil.Process().memory_info()
        metadata["process_peak_working_set_bytes"] = getattr(memory, "peak_wset", None)
        metadata["process_final_resident_bytes"] = memory.rss
        metadata["ended_at_utc"] = datetime.now(UTC).isoformat()
        metadata["duration_seconds"] = time.perf_counter() - started
        metadata_path_out.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metadata
