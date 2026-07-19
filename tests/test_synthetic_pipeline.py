from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from synthdet.data.leakage import validate_leakage
from synthdet.synthetic.contracts import load_synthetic_config, sha256_file
from synthdet.synthetic.generator import (
    _place_object,
    generate_synthetic_pool,
    verify_synthetic_reproduction,
)
from synthdet.synthetic.geometry import (
    box_iou,
    foreground_mask_bbox,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from synthdet.synthetic.object_bank import build_object_bank
from synthdet.synthetic.validation import validate_object_bank, validate_synthetic_pool


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture_config(root: Path, split_identity: str) -> Path:
    config = {
        "synthetic_dataset": {
            "name": "fixture-v1",
            "generator_version": "copy-paste-v1",
            "mode": "distribution_matched_copy_paste",
            "root_seed": 42,
            "full_pool_size": 427,
            "smoke_pool_size": 2,
            "active_split_directory": "manifests/v2",
            "active_split_identity": split_identity,
            "output_directory": "datasets/processed/aquarium/synthetic/v1",
            "manifest_directory": "manifests/synthetic/v1",
            "report_directory": "reports/synthetic",
            "class_names": ["fish", "shark"],
        },
        "object_bank": {
            "extraction_method": "grabcut_bbox_init",
            "fallback_method": "feathered_rectangle",
            "grabcut_iterations": 1,
            "context_fraction": 0.2,
            "minimum_crop_dimension": 3,
            "minimum_crop_area": 9,
            "maximum_mask_coverage": 0.98,
            "minimum_mask_coverage": 0.02,
            "fallback_feather_pixels": 2,
            "contact_sheet_examples_per_class": 1,
        },
        "sampling": {
            "policy": "real_train_distribution",
            "pasted_objects_per_image": {"minimum": 1, "maximum": 1},
            "prefer_different_source_group": True,
            "require_all_classes": True,
            "allowed_extraction_methods": ["grabcut", "feathered_rectangle_fallback"],
            "minimum_generation_mask_coverage": 0.02,
            "maximum_generation_mask_coverage": 1.0,
            "minimum_foreground_mean_luminance": 0.0,
            "minimum_largest_component_fraction": 0.0,
            "minimum_foreground_bbox_fill": 0.0,
        },
        "transforms": {
            "scale": [0.8, 0.8],
            "rotation_degrees": [0, 0],
            "horizontal_flip_probability": 0,
            "vertical_flip_probability": 0,
            "brightness": [1, 1],
            "contrast": [1, 1],
            "saturation": [1, 1],
            "blur_probability": 0,
            "blur_radius": [0, 0],
            "noise_probability": 0,
            "noise_standard_deviation": [0, 0],
            "jpeg_quality": [90, 90],
            "alpha_feather_radius": [0.5, 0.5],
        },
        "placement": {
            "minimum_dimension_pixels": 3,
            "maximum_image_fraction": 0.45,
            "maximum_iou": 0.1,
            "maximum_existing_object_occlusion": 0.1,
            "maximum_retries_per_object": 100,
            "maximum_sample_attempts": 4,
            "foreground_alpha_threshold": 16,
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _create_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_dir = root / "manifests" / "v2"
    manifest_fields = [
        "image_path",
        "label_path",
        "content_hash",
        "perceptual_hash",
        "source_group_id",
        "image_width",
        "image_height",
        "classes_present",
        "object_count",
        "split",
        "inclusion_status",
    ]
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    duplicate_rows: list[dict[str, str]] = []
    specifications = [
        ("train", 0, "fish", "red", "source-a"),
        ("train", 1, "shark", "green", "source-b"),
        ("val", 0, "fish", "blue", "source-c"),
        ("test", 1, "shark", "purple", "source-d"),
    ]
    for index, (split, class_id, class_name, color, group) in enumerate(specifications):
        image_relative = f"datasets/raw/{split}/images/image-{index}.jpg"
        label_relative = f"datasets/raw/{split}/labels/image-{index}.txt"
        image_path = root / image_relative
        label_path = root / label_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (80, 80), "gray")
        draw = ImageDraw.Draw(image)
        draw.rectangle((28, 28, 51, 51), fill=color)
        image.save(image_path, quality=95)
        label_path.write_text(f"{class_id} 0.5 0.5 0.3 0.3\n", encoding="utf-8")
        rows_by_split[split].append(
            {
                "image_path": image_relative,
                "label_path": label_relative,
                "content_hash": sha256_file(image_path),
                "perceptual_hash": f"{index:016x}",
                "source_group_id": group,
                "image_width": "80",
                "image_height": "80",
                "classes_present": class_name,
                "object_count": "1",
                "split": split,
                "inclusion_status": "included",
            }
        )
        duplicate_rows.append(
            {
                "image_path": image_relative,
                "content_hash": sha256_file(image_path),
                "perceptual_hash": f"{index:016x}",
                "duplicate_group_id": "",
                "match_type": "unique",
                "minimum_hamming_distance": "",
                "review_status": "not_applicable",
            }
        )
    hashes: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        path = split_dir / f"real_{split}.csv"
        _write_csv(path, manifest_fields, rows)
        hashes[path.name] = sha256_file(path)
    _write_csv(split_dir / "duplicate_groups.csv", list(duplicate_rows[0]), duplicate_rows)
    hashes["duplicate_groups.csv"] = sha256_file(split_dir / "duplicate_groups.csv")
    _write_csv(
        split_dir / "excluded.csv",
        ["image_path", "label_path", "exclusion_reasons", "inclusion_status"],
        [],
    )
    hashes["excluded.csv"] = sha256_file(split_dir / "excluded.csv")
    material = "\n".join(f"{name}:{hashes[name]}" for name in sorted(hashes)).encode()
    metadata = {
        "combined_split_sha256": hashlib.sha256(material).hexdigest(),
        "manifest_sha256": hashes,
    }
    (split_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    config_path = _fixture_config(root, metadata["combined_split_sha256"])
    output = root / "datasets/processed/aquarium/synthetic/v1"
    manifests = root / "manifests/synthetic/v1"
    return config_path, output, manifests


def test_box_conversion_and_mask_bbox() -> None:
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 80)
    assert box == (40, 24, 60, 56)
    assert xyxy_to_yolo(box, 100, 80) == pytest.approx((0.5, 0.5, 0.2, 0.4))
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    assert foreground_mask_bbox(mask) == (3, 2, 9, 8)


def test_iou_rejection_geometry() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert box_iou((0, 0, 5, 5), (6, 6, 10, 10)) == 0


def test_placement_stays_in_bounds_and_rejects_excessive_overlap(tmp_path: Path) -> None:
    config = load_synthetic_config(_fixture_config(tmp_path, "a" * 64))
    canvas = Image.new("RGB", (80, 80), "gray")
    foreground = Image.new("RGB", (12, 12), "red")
    alpha = np.full((12, 12), 255, dtype=np.uint8)
    placed = _place_object(
        canvas,
        foreground,
        alpha,
        [],
        config,
        np.random.default_rng(42),
    )
    assert placed is not None
    x1, y1, x2, y2 = placed[0]
    assert 0 <= x1 < x2 <= canvas.width
    assert 0 <= y1 < y2 <= canvas.height

    config.placement["maximum_iou"] = 0.0
    blocked = _place_object(
        canvas,
        foreground,
        alpha,
        [(0, 0, 80, 80)],
        config,
        np.random.default_rng(42),
    )
    assert blocked is None


def test_train_only_object_bank_and_deterministic_generation(tmp_path: Path) -> None:
    config_path, output, manifests = _create_fixture(tmp_path)
    config = load_synthetic_config(config_path)
    summary = build_object_bank(
        config,
        tmp_path,
        output,
        Path(config.dataset["output_directory"]),
        manifests,
    )
    assert summary["included_objects"] == 2
    assert validate_object_bank(config, tmp_path, output, manifests) == []

    first = generate_synthetic_pool(
        config,
        tmp_path,
        output,
        Path(config.dataset["output_directory"]),
        manifests,
        manifests,
        output,
        2,
        ensure_class_coverage=True,
    )
    assert first["generated_count"] == 2
    assert (
        validate_synthetic_pool(
            config,
            tmp_path,
            output,
            Path(config.dataset["output_directory"]),
            manifests,
            manifests,
            2,
        )
        == []
    )
    synthetic_rows = list(csv.DictReader((manifests / "synthetic_images.csv").open()))
    for row in synthetic_rows:
        label = output / Path(row["synthetic_label_path"]).relative_to(
            Path(config.dataset["output_directory"])
        )
        assert len(label.read_text(encoding="utf-8").splitlines()) == 2

    reproduced = verify_synthetic_reproduction(config, tmp_path, output, manifests)
    assert (
        reproduced["combined_synthetic_pool_identity"] == first["combined_synthetic_pool_identity"]
    )
    with pytest.raises(FileExistsError):
        generate_synthetic_pool(
            config,
            tmp_path,
            output,
            Path(config.dataset["output_directory"]),
            manifests,
            manifests,
            output,
            2,
        )


def test_validation_rejects_protected_source(tmp_path: Path) -> None:
    config_path, output, manifests = _create_fixture(tmp_path)
    config = load_synthetic_config(config_path)
    build_object_bank(
        config,
        tmp_path,
        output,
        Path(config.dataset["output_directory"]),
        manifests,
    )
    rows = list(csv.DictReader((manifests / "object_bank.csv").open()))
    validation_row = list(csv.DictReader((tmp_path / "manifests/v2/real_val.csv").open()))[0]
    rows[0]["image_path"] = validation_row["image_path"]
    rows[0]["content_hash"] = validation_row["content_hash"]
    _write_csv(manifests / "object_bank.csv", list(rows[0]), rows)
    errors = validate_object_bank(config, tmp_path, output, manifests)
    assert any("not real_train" in error for error in errors)


def test_leakage_hard_fails_on_validation_background(tmp_path: Path) -> None:
    _create_fixture(tmp_path)
    validation_row = list(csv.DictReader((tmp_path / "manifests/v2/real_val.csv").open()))[0]
    future = tmp_path / "backgrounds.csv"
    _write_csv(
        future,
        ["image_path", "content_hash"],
        [
            {
                "image_path": validation_row["image_path"],
                "content_hash": validation_row["content_hash"],
            }
        ],
    )
    errors = validate_leakage(tmp_path / "manifests/v2", synthetic_background_manifests=[future])
    assert any("Validation image path" in error for error in errors)
