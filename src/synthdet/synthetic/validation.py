"""Hard-fail validation of object-bank and synthetic-pool artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from synthdet.data.leakage import validate_leakage
from synthdet.synthetic.contracts import (
    SyntheticConfig,
    read_csv,
    sha256_file,
    stable_json_hash,
)


def _actual_output_path(
    canonical: str, logical_output_root: Path, physical_output_root: Path
) -> Path:
    return physical_output_root / Path(canonical).relative_to(logical_output_root)


def validate_object_bank(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    manifest_dir: Path,
) -> list[str]:
    errors: list[str] = []
    train_rows = read_csv(
        project_root / Path(config.dataset["active_split_directory"]) / "real_train.csv"
    )
    train_by_path = {row["image_path"]: row for row in train_rows}
    rows = read_csv(manifest_dir / "object_bank.csv")
    excluded_rows = read_csv(manifest_dir / "excluded_objects.csv")
    included = [row for row in rows if row["inclusion_status"] == "included"]
    excluded = [row for row in rows if row["inclusion_status"] == "excluded"]
    if excluded != excluded_rows:
        errors.append("excluded_objects.csv is not the exact excluded object-bank subset")
    object_ids = [row["object_id"] for row in rows]
    if len(object_ids) != len(set(object_ids)):
        errors.append("Object-bank IDs are not unique")
    logical_root = Path(config.dataset["output_directory"])
    for row in rows:
        if row["image_path"] not in train_by_path:
            errors.append(f"Object-bank source is not real_train: {row['image_path']}")
        elif row["content_hash"] != train_by_path[row["image_path"]]["content_hash"]:
            errors.append(f"Object-bank source hash mismatch: {row['image_path']}")
        if row["inclusion_status"] != "included":
            if not row["exclusion_reason"]:
                errors.append(f"Excluded object lacks a reason: {row['object_id']}")
            continue
        try:
            crop = _actual_output_path(row["crop_output_path"], logical_root, physical_output_root)
            mask = _actual_output_path(row["mask_output_path"], logical_root, physical_output_root)
        except ValueError:
            errors.append(f"Object output path is not canonical: {row['object_id']}")
            continue
        if not crop.is_file() or not mask.is_file():
            errors.append(f"Object crop/mask pair is missing: {row['object_id']}")
            continue
        if sha256_file(crop) != row["crop_hash"] or sha256_file(mask) != row["mask_hash"]:
            errors.append(f"Object crop/mask hash mismatch: {row['object_id']}")
        try:
            with Image.open(crop) as crop_image, Image.open(mask) as mask_image:
                if crop_image.size != mask_image.size:
                    errors.append(f"Object crop and mask dimensions differ: {row['object_id']}")
        except OSError:
            errors.append(f"Object crop or mask cannot be read: {row['object_id']}")
    observed_classes = {row["class_name"] for row in included}
    if observed_classes != set(config.class_names):
        errors.append("Included object bank does not cover all configured classes")
    return errors


def validate_synthetic_pool(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    manifest_dir: Path,
    bank_manifest_dir: Path,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    split_dir = project_root / Path(config.dataset["active_split_directory"])
    train_rows = read_csv(split_dir / "real_train.csv")
    train_by_path = {row["image_path"]: row for row in train_rows}
    protected_rows = read_csv(split_dir / "real_val.csv") + read_csv(split_dir / "real_test.csv")
    protected_paths = {row["image_path"] for row in protected_rows}
    protected_hashes = {row["content_hash"] for row in protected_rows}
    images = read_csv(manifest_dir / "synthetic_images.csv")
    sources = read_csv(manifest_dir / "synthetic_sources.csv")
    backgrounds = read_csv(manifest_dir / "synthetic_backgrounds.csv")
    bank_rows = read_csv(bank_manifest_dir / "object_bank.csv")
    bank_by_id = {
        row["object_id"]: row for row in bank_rows if row["inclusion_status"] == "included"
    }
    if len(images) != expected_count:
        errors.append(f"Synthetic image count is {len(images)}; expected {expected_count}")
    image_paths = [row["synthetic_image_path"] for row in images]
    if len(image_paths) != len(set(image_paths)):
        errors.append("Synthetic image paths are not unique")
    if len(backgrounds) != len(images):
        errors.append("Synthetic background manifest must contain exactly one row per image")
    source_count_by_image = Counter(row["synthetic_image_path"] for row in sources)
    background_by_image = {row["synthetic_image_path"]: row for row in backgrounds}
    observed_class_ids: set[int] = set()
    for row in images:
        canonical_image = row["synthetic_image_path"]
        canonical_label = row["synthetic_label_path"]
        try:
            image_path = _actual_output_path(
                canonical_image, logical_output_root, physical_output_root
            )
            label_path = _actual_output_path(
                canonical_label, logical_output_root, physical_output_root
            )
        except ValueError:
            errors.append(f"Synthetic output path is not canonical: {canonical_image}")
            continue
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"Synthetic image/label pair is missing: {canonical_image}")
            continue
        if sha256_file(image_path) != row["output_image_hash"]:
            errors.append(f"Synthetic image hash mismatch: {canonical_image}")
        if sha256_file(label_path) != row["output_label_hash"]:
            errors.append(f"Synthetic label hash mismatch: {canonical_label}")
        with Image.open(image_path) as image:
            width, height = image.size
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(label_lines) != int(row["total_object_count"]):
            errors.append(f"Synthetic label count mismatch: {canonical_label}")
        if int(row["pasted_object_count"]) < 1:
            errors.append(f"Synthetic image has no pasted objects: {canonical_image}")
        if source_count_by_image[canonical_image] != int(row["pasted_object_count"]):
            errors.append(f"Synthetic source count mismatch: {canonical_image}")
        for line in label_lines:
            values = line.split()
            if len(values) != 5:
                errors.append(f"Invalid YOLO field count: {canonical_label}")
                continue
            class_id = int(values[0])
            normalized = [float(value) for value in values[1:]]
            if not 0 <= class_id < len(config.class_names):
                errors.append(f"Unsupported generated class ID: {canonical_label}")
            elif (
                not all(0 <= value <= 1 for value in normalized)
                or normalized[2] <= 0
                or normalized[3] <= 0
            ):
                errors.append(f"Invalid normalized generated box: {canonical_label}")
            else:
                observed_class_ids.add(class_id)
        background = background_by_image.get(canonical_image)
        if background is None:
            errors.append(f"Synthetic image has no background provenance: {canonical_image}")
            continue
        base = train_by_path.get(background["image_path"])
        if base is None or background["image_path"] in protected_paths:
            errors.append(f"Synthetic background is not train-only: {canonical_image}")
        elif background["content_hash"] != base["content_hash"]:
            errors.append(f"Synthetic background hash mismatch: {canonical_image}")
        base_lines = (
            (project_root / background["base_label_path"]).read_text(encoding="utf-8").splitlines()
        )
        if len(base_lines) != int(row["retained_base_object_count"]):
            errors.append(f"Retained base-label count mismatch: {canonical_image}")
        generated_prefix = label_lines[: len(base_lines)]
        for original, generated in zip(base_lines, generated_prefix, strict=True):
            original_values = [float(value) for value in original.split()]
            generated_values = [float(value) for value in generated.split()]
            if any(
                abs(first - second) > 1e-9
                for first, second in zip(original_values, generated_values, strict=True)
            ):
                errors.append(f"Base annotation was not preserved: {canonical_image}")
                break
        if int(row["image_width"]) != width or int(row["image_height"]) != height:
            errors.append(f"Synthetic image dimensions disagree with manifest: {canonical_image}")
        if row["active_real_split_identity"] != config.split_identity:
            errors.append(f"Synthetic row has the wrong real split identity: {canonical_image}")
    for row in sources:
        bank = bank_by_id.get(row["object_id"])
        if bank is None:
            errors.append(f"Synthetic source object is absent from usable bank: {row['object_id']}")
            continue
        if row["image_path"] != bank["image_path"] or row["content_hash"] != bank["content_hash"]:
            errors.append(f"Synthetic object provenance mismatch: {row['object_id']}")
        if row["image_path"] not in train_by_path or row["image_path"] in protected_paths:
            errors.append(f"Synthetic object source is not train-only: {row['object_id']}")
        if row["content_hash"] in protected_hashes:
            errors.append(f"Protected content hash used as an object source: {row['object_id']}")
    if observed_class_ids != set(range(len(config.class_names))):
        errors.append("Synthetic pool does not contain every configured class")
    actual_images = sorted(path.name for path in (physical_output_root / "images").glob("*.jpg"))
    actual_labels = sorted(path.name for path in (physical_output_root / "labels").glob("*.txt"))
    if len(actual_images) != len(images) or len(actual_labels) != len(images):
        errors.append("Materialized image/label file counts do not match the manifest")
    if [Path(name).stem for name in actual_images] != [Path(name).stem for name in actual_labels]:
        errors.append("Materialized image/label basenames do not match")
    data_yaml_path = physical_output_root / "data.yaml"
    if not data_yaml_path.is_file():
        errors.append("Materialized YOLO data.yaml is missing")
    else:
        data_yaml: dict[str, Any] = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        names = data_yaml.get("names", {})
        normalized_names = (
            [names[index] for index in sorted(names)] if isinstance(names, dict) else names
        )
        if normalized_names != config.class_names:
            errors.append("Materialized data.yaml class order is incorrect")
        if data_yaml.get("train") != "images" or data_yaml.get("val") is not None:
            errors.append("Materialized data.yaml is not an explicit train-only synthetic view")
    errors.extend(
        validate_leakage(
            split_dir,
            synthetic_source_manifests=[manifest_dir / "synthetic_sources.csv"],
            synthetic_background_manifests=[manifest_dir / "synthetic_backgrounds.csv"],
            synthetic_image_manifests=[manifest_dir / "synthetic_images.csv"],
            expected_split_identity=config.split_identity,
        )
    )
    metadata_path = manifest_dir / "generation_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hashes: dict[str, str] = {}
        for name, expected in metadata.get("output_manifest_hashes", {}).items():
            if name == "materialized_data.yaml":
                actual = sha256_file(data_yaml_path)
            else:
                candidate = manifest_dir / name
                if not candidate.is_file():
                    candidate = bank_manifest_dir / name
                actual = sha256_file(candidate)
            actual_hashes[name] = actual
            if actual != expected:
                errors.append(f"Synthetic manifest hash mismatch: {name}")
        if stable_json_hash(actual_hashes) != metadata.get("combined_synthetic_pool_identity"):
            errors.append("Combined synthetic-pool identity mismatch")
    else:
        errors.append("generation_metadata.json is missing")
    output_hashes = [row["output_image_hash"] for row in images]
    if len(output_hashes) != len(set(output_hashes)):
        errors.append("Exact duplicate images exist in the synthetic outputs")
    real_hashes = {
        row["content_hash"]
        for split in ("train", "val", "test")
        for row in read_csv(split_dir / f"real_{split}.csv")
    }
    if real_hashes & set(output_hashes):
        errors.append("A synthetic output image exactly collides with a real image")
    return errors
