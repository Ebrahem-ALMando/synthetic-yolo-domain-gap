from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from synthdet.synthetic.contracts import sha256_file, stable_json_hash, write_csv
from synthdet.training.experiments import (
    MANIFEST_FIELDS,
    construct_regimes,
    deterministic_multilabel_select,
    freeze_experiment_design,
    validate_regimes,
    verify_experiment_reproduction,
)
from synthdet.training.materialize import materialize_views, validate_materialized_views
from synthdet.training.runner import create_run_directory, run_training, validate_runner_inputs

CLASS_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
SMALL_COUNTS = {
    "synthetic_only": (0, 8),
    "real_25": (2, 6),
    "real_50": (4, 4),
    "real_75": (6, 2),
    "real_only": (8, 0),
}


def _fixture_rows(root: Path):
    real: list[dict[str, str]] = []
    synthetic: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for index in range(10):
        split = "train" if index < 8 else "val" if index == 8 else "test"
        image_relative = f"datasets/{split}/image-{index}.jpg"
        label_relative = f"datasets/{split}/image-{index}.txt"
        image = root / image_relative
        label = root / label_relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (index * 20, 40, 80)).save(image)
        class_id = index % len(CLASS_NAMES)
        label.write_text(f"{class_id} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        row = {
            "image_path": image_relative,
            "label_path": label_relative,
            "content_hash": sha256_file(image),
            "source_group_id": f"group-{index // 2}",
            "classes_present": CLASS_NAMES[class_id],
        }
        if split == "train":
            real.append(row)
            synthetic_image_relative = f"datasets/synthetic/image-{index}.jpg"
            synthetic_label_relative = f"datasets/synthetic/image-{index}.txt"
            synthetic_image = root / synthetic_image_relative
            synthetic_label = root / synthetic_label_relative
            synthetic_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (index * 20, 80, 40)).save(synthetic_image)
            synthetic_label.write_text(label.read_text(encoding="utf-8"), encoding="utf-8")
            synthetic.append(
                {
                    "synthetic_image_path": synthetic_image_relative,
                    "synthetic_label_path": synthetic_label_relative,
                    "output_image_hash": sha256_file(synthetic_image),
                    "output_label_hash": sha256_file(synthetic_label),
                    "base_image_path": image_relative,
                    "base_source_group_id": row["source_group_id"],
                    "class_ids_present": str(class_id),
                }
            )
        elif split == "val":
            validation.append(row)
        else:
            test.append(row)
    return real, synthetic, validation, test


def _regimes(root: Path):
    real, synthetic, validation, test = _fixture_rows(root)
    regimes = construct_regimes(
        real, synthetic, CLASS_NAMES, root, 42, "r" * 64, "s" * 64, SMALL_COUNTS
    )
    return regimes, real, synthetic, validation, test


def test_exact_counts_complementary_pairing_and_determinism(tmp_path: Path) -> None:
    regimes, real, synthetic, _, _ = _regimes(tmp_path)
    repeated = construct_regimes(
        real, synthetic, CLASS_NAMES, tmp_path, 42, "r" * 64, "s" * 64, SMALL_COUNTS
    )
    assert regimes == repeated
    expected = {row["image_path"] for row in real}
    for name, rows in regimes.items():
        assert len(rows) == 8
        assert len({row["underlying_real_canvas_path"] for row in rows}) == 8
        assert {row["underlying_real_canvas_path"] for row in rows} == expected
        assert sum(row["sample_type"] == "real" for row in rows) == SMALL_COUNTS[name][0]


def test_validation_rejects_test_content(tmp_path: Path) -> None:
    regimes, real, synthetic, validation, test = _regimes(tmp_path)
    assert (
        validate_regimes(
            regimes,
            real,
            validation,
            test,
            synthetic,
            tmp_path,
            "r" * 64,
            "s" * 64,
            SMALL_COUNTS,
        )
        == []
    )
    regimes["real_only"][0]["training_image_path"] = test[0]["image_path"]
    regimes["real_only"][0]["image_hash"] = test[0]["content_hash"]
    errors = validate_regimes(
        regimes,
        real,
        validation,
        test,
        synthetic,
        tmp_path,
        "r" * 64,
        "s" * 64,
        SMALL_COUNTS,
    )
    assert any("protected image" in error for error in errors)


def test_multilabel_selector_preserves_coverage_when_practical() -> None:
    rows = [{"path": str(index), "class_ids_present": str(index % 7)} for index in range(14)]
    selected = deterministic_multilabel_select(rows, 7, CLASS_NAMES, 42, "path")
    classes = {int(row["class_ids_present"]) for row in rows if row["path"] in selected}
    assert classes == set(range(7))


def test_materialized_pairing_annotations_class_order_and_validation_identity(
    tmp_path: Path,
) -> None:
    regimes, _, _, validation, _ = _regimes(tmp_path)
    views = tmp_path / "views"
    materialize_views(regimes, validation, views, tmp_path, CLASS_NAMES, 42, 7, 1)
    errors = validate_materialized_views(
        views,
        CLASS_NAMES,
        {row["content_hash"] for row in validation},
        8,
        1,
        SMALL_COUNTS,
    )
    assert errors == []
    assert yaml.safe_load((views / "real_only/data.yaml").read_text())["names"][0] == "fish"


def test_freeze_no_overwrite_and_identity_reproduction(tmp_path: Path) -> None:
    regimes, real, synthetic, _, _ = _regimes(tmp_path)
    frozen = tmp_path / "frozen"
    metadata = freeze_experiment_design(
        frozen,
        regimes,
        "r" * 64,
        "s" * 64,
        "o" * 64,
        "g" * 64,
        42,
        1,
        Path.cwd(),
        [],
    )
    reproduced = verify_experiment_reproduction(frozen, real, synthetic, CLASS_NAMES, tmp_path, 42)
    assert (
        reproduced["combined_experiment_design_identity"]
        == metadata["combined_experiment_design_identity"]
    )
    with pytest.raises(FileExistsError):
        freeze_experiment_design(
            frozen, regimes, "r" * 64, "s" * 64, "o" * 64, "g" * 64, 42, 1, Path.cwd(), []
        )


def _runner_fixture(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    view = tmp_path / "view"
    (view / "train/images").mkdir(parents=True)
    (view / "train/labels").mkdir(parents=True)
    (view / "val/images").mkdir(parents=True)
    (view / "val/labels").mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(view / "train/images/train.jpg")
    (view / "train/labels/train.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    Image.new("RGB", (16, 16), "green").save(view / "val/images/val.jpg")
    (view / "val/labels/val.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    train_image = view / "train/images/train.jpg"
    train_label = view / "train/labels/train.txt"
    write_csv(
        manifest,
        MANIFEST_FIELDS,
        [
            {
                "regime_name": "real_only",
                "training_image_path": train_image.relative_to(tmp_path).as_posix(),
                "label_path": train_label.relative_to(tmp_path).as_posix(),
                "sample_type": "real",
                "underlying_real_canvas_path": train_image.relative_to(tmp_path).as_posix(),
                "source_group_id": "fixture",
                "image_hash": sha256_file(train_image),
                "label_hash": sha256_file(train_label),
                "class_ids_present": "0",
                "selection_seed": "42",
                "real_split_identity": "r" * 64,
                "synthetic_pool_identity": "s" * 64,
            }
        ],
    )
    data = {"train": "train/images", "val": "val/images", "names": dict(enumerate(CLASS_NAMES))}
    (view / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest_hash = sha256_file(manifest)
    regime = {
        "regime": "real_only",
        "manifest": manifest.relative_to(tmp_path).as_posix(),
        "dataset_view": view.relative_to(tmp_path).as_posix(),
        "expected_manifest_hash": manifest_hash,
    }
    common = {
        "identities": {"experiment_design": "e" * 64},
        "dataset": {"class_names": CLASS_NAMES, "train_budget": 1, "validation_count": 1},
        "training": {"epochs": 1},
        "augmentation": {},
        "smoke": {"view_root": "view", "overrides": {}},
        "outputs": {"run_root": "runs"},
        "model": {"architecture": "fixture", "pretrained_weights": "fake.pt"},
        "hardware_profiles": {
            "standard": {"batch": 16, "imgsz": 640, "workers": 0},
            "low_memory": {"batch": 4, "imgsz": 640, "workers": 0},
        },
    }
    metadata = {
        "combined_experiment_design_identity": "e" * 64,
        "regime_manifest_hashes": {manifest.name: manifest_hash},
    }
    return regime, common, metadata


def test_runner_dry_run_validation(tmp_path: Path) -> None:
    regime, common, metadata = _runner_fixture(tmp_path)
    validated = validate_runner_inputs(
        regime, common, metadata, tmp_path, "final", verify_project=False
    )
    assert validated["regime"] == "real_only"


def test_runner_rejects_missing_or_windows_absolute_regime_yaml(tmp_path: Path) -> None:
    regime, common, metadata = _runner_fixture(tmp_path)
    data_path = tmp_path / "view/data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["train"] = "C:\\datasets\\train\\images"
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong train path"):
        validate_runner_inputs(regime, common, metadata, tmp_path, "final", verify_project=False)
    data_path.unlink()
    with pytest.raises(FileNotFoundError, match="not materialized"):
        validate_runner_inputs(regime, common, metadata, tmp_path, "final", verify_project=False)


def test_failed_run_records_status_and_smoke_metadata(tmp_path: Path, monkeypatch) -> None:
    regime, common, metadata = _runner_fixture(tmp_path)
    regime_path = tmp_path / "regime.yaml"
    common_path = tmp_path / "common.yaml"
    metadata_path = tmp_path / "metadata.json"
    regime_path.write_text(yaml.safe_dump(regime), encoding="utf-8")
    common_path.write_text(yaml.safe_dump(common), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "fake.pt").write_bytes(b"fixture weight")
    profile_inputs = {
        "profile_name": "standard",
        "batch": 16,
        "imgsz": 640,
        "device": "cpu",
        "base_weight_sha256": sha256_file(tmp_path / "fake.pt"),
        "identities": common["identities"],
        "preflight": [],
    }
    profile = {
        **profile_inputs,
        "status": "frozen",
        "environment": {},
        "profile_identity": stable_json_hash(profile_inputs),
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    class BrokenYOLO:
        def __init__(self, _: str):
            pass

        def train(self, **_: object):
            raise RuntimeError("fixture failure")

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=BrokenYOLO))
    monkeypatch.setattr("synthdet.training.runner._git_revision", lambda _: "f" * 40)
    monkeypatch.setattr(
        "synthdet.training.runner.collect_environment", lambda _: {"classification": "fixture"}
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        run_training(
            regime_path,
            common_path,
            metadata_path,
            tmp_path,
            "final",
            "cpu",
            ["fixture"],
            profile_path=profile_path,
            verify_project=False,
        )
    record = json.loads(next((tmp_path / "runs/final").glob("*/run_metadata.json")).read_text())
    assert record["status"] == "failed"
    assert record["test_set_used"] is False
    assert record["ended_at_utc"]


def test_unique_run_directory_never_overwrites(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, "real_only", "smoke")
    second = create_run_directory(tmp_path, "real_only", "smoke")
    assert first != second
    assert first.is_dir() and second.is_dir()
