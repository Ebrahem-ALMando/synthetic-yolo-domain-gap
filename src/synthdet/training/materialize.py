"""Materialize and validate ignored Ultralytics-compatible experiment views."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from synthdet.synthetic.contracts import sha256_file
from synthdet.training.experiments import REGIME_COUNTS, deterministic_multilabel_select


def validate_yolo_label(path: Path, class_count: int) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(f"{path}:{line_number}: expected five fields")
            continue
        try:
            class_id = int(values[0])
            box = [float(value) for value in values[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric annotation")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number}: invalid class ID")
        if not all(0.0 <= value <= 1.0 for value in box) or box[2] <= 0 or box[3] <= 0:
            errors.append(f"{path}:{line_number}: invalid normalized box")
    return errors


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise FileExistsError(f"Refusing to overwrite different file: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialize_rows(
    rows: list[dict[str, str]], target_root: Path, project_root: Path, prefix: str
) -> dict[str, int]:
    methods: dict[str, int] = {}
    for index, row in enumerate(rows):
        image = project_root / row["training_image_path"]
        label = project_root / row["label_path"]
        stem = f"{prefix}-{index:04d}-{row['image_hash'][:12]}"
        method = _link_or_copy(image, target_root / "images" / f"{stem}{image.suffix.lower()}")
        _link_or_copy(label, target_root / "labels" / f"{stem}.txt")
        methods[method] = methods.get(method, 0) + 1
    return methods


def _validation_as_training_rows(
    rows: list[dict[str, str]], project_root: Path
) -> list[dict[str, str]]:
    return [
        {
            "training_image_path": row["image_path"],
            "label_path": row["label_path"],
            "image_hash": row["content_hash"],
            "class_ids_present": "",
        }
        for row in rows
    ]


def _write_data_yaml(root: Path, class_names: list[str]) -> None:
    data = {
        "path": root.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _smoke_select(
    rows: list[dict[str, str]], count: int, class_names: list[str], seed: int
) -> list[dict[str, str]]:
    selected = deterministic_multilabel_select(
        rows, min(count, len(rows)), class_names, seed, "training_image_path"
    )
    return [row for row in rows if row["training_image_path"] in selected]


def materialize_views(
    regimes: dict[str, list[dict[str, str]]],
    validation_rows: list[dict[str, str]],
    output_root: Path,
    project_root: Path,
    class_names: list[str],
    seed: int,
    smoke_train_count: int = 16,
    smoke_val_count: int = 14,
) -> dict[str, Any]:
    validation = _validation_as_training_rows(validation_rows, project_root)
    for row, source in zip(validation, validation_rows, strict=True):
        row["class_ids_present"] = ";".join(
            str(class_names.index(name)) for name in source["classes_present"].split(";") if name
        )
    summary: dict[str, Any] = {}
    smoke_validation = _smoke_select(validation, smoke_val_count, class_names, seed)
    for regime, rows in regimes.items():
        regime_root = output_root / regime
        methods = _materialize_rows(rows, regime_root / "train", project_root, "train")
        _materialize_rows(validation, regime_root / "val", project_root, "val")
        _write_data_yaml(regime_root, class_names)
        smoke_root = output_root / "_smoke" / regime
        smoke_rows = _smoke_select(rows, smoke_train_count, class_names, seed)
        _materialize_rows(smoke_rows, smoke_root / "train", project_root, "train")
        _materialize_rows(smoke_validation, smoke_root / "val", project_root, "val")
        _write_data_yaml(smoke_root, class_names)
        summary[regime] = {
            "train_count": len(rows),
            "validation_count": len(validation),
            "smoke_train_count": len(smoke_rows),
            "smoke_validation_count": len(smoke_validation),
            "materialization_methods": methods,
        }
    (output_root / "materialization_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def validate_materialized_views(
    output_root: Path,
    class_names: list[str],
    expected_validation_hashes: set[str],
    expected_train_count: int = 427,
    expected_validation_count: int = 140,
    regime_counts: dict[str, tuple[int, int]] = REGIME_COUNTS,
) -> list[str]:
    errors: list[str] = []
    validation_identities: set[tuple[str, ...]] = set()
    for regime in regime_counts:
        root = output_root / regime
        data_path = root / "data.yaml"
        if not data_path.is_file():
            errors.append(f"{regime}: data.yaml is missing")
            continue
        data: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        names = data.get("names", {})
        normalized = [names[index] for index in sorted(names)] if isinstance(names, dict) else names
        if normalized != class_names:
            errors.append(f"{regime}: class order mismatch")
        train_images = sorted((root / "train/images").glob("*"))
        train_labels = sorted((root / "train/labels").glob("*.txt"))
        val_images = sorted((root / "val/images").glob("*"))
        val_labels = sorted((root / "val/labels").glob("*.txt"))
        if len(train_images) != expected_train_count or len(train_labels) != expected_train_count:
            errors.append(f"{regime}: training pair count is not {expected_train_count}")
        if (
            len(val_images) != expected_validation_count
            or len(val_labels) != expected_validation_count
        ):
            errors.append(f"{regime}: validation pair count is not {expected_validation_count}")
        if {path.stem for path in train_images} != {path.stem for path in train_labels}:
            errors.append(f"{regime}: training image-label stems differ")
        if {path.stem for path in val_images} != {path.stem for path in val_labels}:
            errors.append(f"{regime}: validation image-label stems differ")
        for label in train_labels + val_labels:
            errors.extend(validate_yolo_label(label, len(class_names)))
        validation_hashes = tuple(sorted(sha256_file(path) for path in val_images))
        validation_identities.add(validation_hashes)
        if set(validation_hashes) != expected_validation_hashes:
            errors.append(f"{regime}: validation identity differs from active real validation")
    if len(validation_identities) != 1:
        errors.append("Validation views are not identical across all regimes")
    return errors
