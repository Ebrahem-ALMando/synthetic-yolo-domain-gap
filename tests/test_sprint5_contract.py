from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from synthdet.evaluation.contract import REGIMES, TIE_BREAKERS, validate_contract
from synthdet.training.intake import CLASS_NAMES


def _fixture(root: Path) -> Path:
    manifest = root / "manifests/aquarium/v2/real_test.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("image_path,label_path\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = {
        "status": "frozen",
        "combined_split_sha256": "b" * 64,
        "actual_counts": {"test": 68},
        "image_count_per_class_per_split": {"test": {"penguin": 4}},
        "manifest_sha256": {"real_test.csv": digest},
    }
    (manifest.parent / "split_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    checkpoints = {}
    for regime in REGIMES:
        checkpoint = root / f"artifacts/{regime}/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(regime.encode())
        checkpoints[regime] = {
            "path": checkpoint.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    contract = {
        "schema_version": 1,
        "status": "frozen",
        "campaign_id": "fixture",
        "contract_source_revision": "a" * 40,
        "training_identity": "c" * 64,
        "protected_test": {
            "manifest_path": manifest.relative_to(root).as_posix(),
            "manifest_sha256": digest,
            "real_split_identity": "b" * 64,
            "expected_image_count": 68,
            "penguin_image_count": 4,
            "authorized_access_campaign_id": "fixture",
            "access_count_before_campaign": 0,
        },
        "class_names": CLASS_NAMES,
        "checkpoints": checkpoints,
        "runtime": {},
        "evaluation": {
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
        },
        "object_size_policy": {
            "coordinate_space": "original_image_pixels",
            "small": {"maximum_area_px_exclusive": 1024},
            "medium": {
                "minimum_area_px_inclusive": 1024,
                "maximum_area_px_exclusive": 9216,
            },
            "large": {"minimum_area_px_inclusive": 9216},
        },
        "ranking": {"primary_metric": "map50_95", "tie_breakers": list(TIE_BREAKERS)},
        "campaign": {
            "expected_complete_models": 5,
            "maximum_successful_campaigns": 1,
            "partial_results_must_not_drive_changes": True,
            "alternative_threshold_runs_forbidden": True,
        },
        "output_schema": {"required_model_metrics": ["map50_95"]},
    }
    path = root / "contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return path


def test_contract_validates_without_reading_protected_pixels(tmp_path: Path) -> None:
    validation = validate_contract(tmp_path, _fixture(tmp_path), check_runtime=False)
    assert validation["test_image_count"] == 68
    assert validation["protected_image_pixels_read"] is False
    assert tuple(validation["checkpoint_hashes"]) == REGIMES


def test_contract_rejects_checkpoint_hash_change(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    data = yaml.safe_load(contract.read_text(encoding="utf-8"))
    data["checkpoints"]["real_50"]["sha256"] = "0" * 64
    contract.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        validate_contract(tmp_path, contract, check_runtime=False)


def test_contract_rejects_ranking_mutation(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    data = yaml.safe_load(contract.read_text(encoding="utf-8"))
    data["ranking"]["primary_metric"] = "map50"
    contract.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="ranking policy"):
        validate_contract(tmp_path, contract, check_runtime=False)
