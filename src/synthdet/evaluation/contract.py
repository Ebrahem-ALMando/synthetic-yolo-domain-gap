"""Sprint 5 evaluation-contract generation and pre-access validation."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml
from ultralytics import __version__ as ultralytics_version

from synthdet.synthetic.contracts import sha256_file
from synthdet.training.intake import CLASS_NAMES

REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
PRIMARY_METRIC = "map50_95"
TIE_BREAKERS = (
    "map50",
    "macro_per_class_ap50_95",
    "recall",
    "latency_ms",
    "checkpoint_size_bytes",
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "evaluation contract")


def _portable_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path must be a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{label} path is not repository-relative and portable")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {value}")
    return path


def validate_contract(
    root: Path,
    contract_path: Path,
    *,
    check_runtime: bool = True,
    check_checkpoints: bool = True,
) -> dict[str, Any]:
    """Validate the frozen contract without reading protected image pixels or labels."""

    contract = load_contract(contract_path)
    if contract.get("schema_version") != 1 or contract.get("status") != "frozen":
        raise ValueError("Sprint 5 contract schema/status is not frozen")
    source_revision = contract.get("contract_source_revision")
    if not isinstance(source_revision, str) or not REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("Contract source revision is invalid")
    training_identity = contract.get("training_identity")
    if not isinstance(training_identity, str) or not HASH_PATTERN.fullmatch(training_identity):
        raise ValueError("Training identity is invalid")

    protected = _mapping(contract.get("protected_test"), "protected test contract")
    if protected.get("expected_image_count") != 68 or protected.get("penguin_image_count") != 4:
        raise ValueError("Protected-test/Penguin counts differ from frozen Split V2")
    if protected.get("authorized_access_campaign_id") != contract.get("campaign_id"):
        raise ValueError("Protected-test authorization campaign mismatch")
    if protected.get("access_count_before_campaign") != 0:
        raise ValueError("Contract does not begin with zero authorized test access")
    manifest = _portable_file(root, protected.get("manifest_path"), "protected-test manifest")
    if sha256_file(manifest) != protected.get("manifest_sha256"):
        raise ValueError("Protected-test manifest SHA-256 mismatch")
    metadata = json.loads(
        (root / "manifests/aquarium/v2/split_metadata.json").read_text(encoding="utf-8")
    )
    if (
        metadata["status"] != "frozen"
        or metadata["combined_split_sha256"] != protected.get("real_split_identity")
        or metadata["actual_counts"]["test"] != 68
        or metadata["image_count_per_class_per_split"]["test"]["penguin"] != 4
        or metadata["manifest_sha256"]["real_test.csv"] != protected.get("manifest_sha256")
    ):
        raise ValueError("Contract protected-test identity differs from Split V2 metadata")

    if contract.get("class_names") != CLASS_NAMES:
        raise ValueError("Contract class order differs from the frozen seven-class order")
    checkpoints = _mapping(contract.get("checkpoints"), "checkpoint registry")
    if tuple(checkpoints) != REGIMES:
        raise ValueError("Contract must list the five regimes in frozen order")
    checkpoint_hashes: dict[str, str] = {}
    for regime in REGIMES:
        checkpoint = _mapping(checkpoints[regime], f"{regime} checkpoint")
        expected_hash = checkpoint.get("sha256")
        if not isinstance(expected_hash, str) or not HASH_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"{regime} checkpoint SHA-256 is invalid")
        if check_checkpoints:
            checkpoint_path = _portable_file(root, checkpoint.get("path"), f"{regime} checkpoint")
            if sha256_file(checkpoint_path) != expected_hash:
                raise ValueError(f"{regime} checkpoint SHA-256 mismatch")
        checkpoint_hashes[regime] = expected_hash

    runtime = _mapping(contract.get("runtime"), "runtime contract")
    expected_runtime = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics_version,
    }
    if check_runtime and any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError(f"Runtime differs from contract: expected {expected_runtime}")

    evaluation = _mapping(contract.get("evaluation"), "shared evaluation configuration")
    expected_evaluation = {
        "split": "test",
        "image_size": 640,
        "batch": 4,
        "device": "cpu",
        "workers": 0,
        "seed": 42,
        "deterministic": True,
        "confidence_threshold": 0.001,
        "iou_threshold": 0.70,
        "max_detections": 300,
        "augment": False,
        "agnostic_nms": False,
        "single_class": False,
        "rectangular_batches": True,
        "half_precision": False,
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise ValueError(f"Shared evaluation configuration differs: {key}")

    size_policy = _mapping(contract.get("object_size_policy"), "object-size policy")
    if (
        size_policy.get("coordinate_space") != "original_image_pixels"
        or size_policy.get("small", {}).get("maximum_area_px_exclusive") != 1024
        or size_policy.get("medium", {}).get("minimum_area_px_inclusive") != 1024
        or size_policy.get("medium", {}).get("maximum_area_px_exclusive") != 9216
        or size_policy.get("large", {}).get("minimum_area_px_inclusive") != 9216
    ):
        raise ValueError("Object-size policy differs from the frozen dataset-audit thresholds")

    ranking = _mapping(contract.get("ranking"), "ranking policy")
    if ranking.get("primary_metric") != PRIMARY_METRIC or tuple(
        ranking.get("tie_breakers", [])
    ) != TIE_BREAKERS:
        raise ValueError("Final ranking policy differs from the preregistered rule")
    campaign = _mapping(contract.get("campaign"), "campaign policy")
    if (
        campaign.get("expected_complete_models") != 5
        or campaign.get("maximum_successful_campaigns") != 1
        or campaign.get("partial_results_must_not_drive_changes") is not True
        or campaign.get("alternative_threshold_runs_forbidden") is not True
    ):
        raise ValueError("Campaign completeness/no-tuning policy is invalid")
    output_schema = _mapping(contract.get("output_schema"), "output schema")
    if not output_schema.get("required_model_metrics"):
        raise ValueError("Contract result schema is empty")

    return {
        "status": "valid",
        "contract_yaml_sha256": sha256_file(contract_path),
        "contract_source_revision": source_revision,
        "campaign_id": contract["campaign_id"],
        "training_identity": training_identity,
        "test_manifest_sha256": protected["manifest_sha256"],
        "test_image_count": 68,
        "checkpoint_hashes": checkpoint_hashes,
        "runtime": runtime,
        "protected_image_pixels_read": False,
        "protected_labels_read": False,
    }


def write_input_contract(
    root: Path, contract_path: Path, output_path: Path, validation: dict[str, Any]
) -> None:
    contract = load_contract(contract_path)
    output = {
        "schema_version": 1,
        "contract_yaml_path": contract_path.relative_to(root).as_posix(),
        "contract_yaml_sha256": validation["contract_yaml_sha256"],
        "contract_source_revision": validation["contract_source_revision"],
        "campaign_id": validation["campaign_id"],
        "training_identity": validation["training_identity"],
        "test_manifest_path": contract["protected_test"]["manifest_path"],
        "test_manifest_sha256": validation["test_manifest_sha256"],
        "test_image_count": validation["test_image_count"],
        "real_split_identity": contract["protected_test"]["real_split_identity"],
        "checkpoints": contract["checkpoints"],
        "class_names": contract["class_names"],
        "runtime": contract["runtime"],
        "evaluation": contract["evaluation"],
        "object_size_policy": contract["object_size_policy"],
        "ranking": contract["ranking"],
        "latency": contract["latency"],
        "campaign": contract["campaign"],
        "output_schema": contract["output_schema"],
        "pre_access_validation": {
            "checkpoint_hashes_verified": True,
            "manifest_hash_verified": True,
            "split_metadata_verified": True,
            "protected_image_pixels_read": False,
            "protected_labels_read": False,
            "test_metrics_observed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
