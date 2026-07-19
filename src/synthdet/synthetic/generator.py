"""Deterministic train-only copy-paste compositing and frozen provenance manifests."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from synthdet.synthetic.contracts import (
    SyntheticConfig,
    derive_seed,
    read_csv,
    sha256_file,
    stable_json_hash,
    write_csv,
)
from synthdet.synthetic.geometry import (
    box_iou,
    existing_object_occlusion,
    foreground_mask_bbox,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from synthdet.synthetic.object_bank import calculate_object_bank_identity

SYNTHETIC_IMAGE_FIELDS = [
    "synthetic_image_path",
    "synthetic_label_path",
    "output_image_hash",
    "output_label_hash",
    "base_image_path",
    "base_image_hash",
    "base_source_group_id",
    "pasted_object_count",
    "retained_base_object_count",
    "class_ids_present",
    "total_object_count",
    "image_width",
    "image_height",
    "per_image_seed",
    "generator_configuration_hash",
    "object_bank_identity",
    "active_real_split_identity",
    "jpeg_quality",
    "generation_status",
]

SOURCE_FIELDS = [
    "synthetic_image_path",
    "object_id",
    "class_id",
    "class_name",
    "image_path",
    "content_hash",
    "source_group_id",
    "duplicate_group_id",
    "scale",
    "rotation_degrees",
    "horizontal_flip",
    "brightness",
    "contrast",
    "saturation",
    "blur_radius",
    "noise_standard_deviation",
    "alpha_feather_radius",
    "placed_x1",
    "placed_y1",
    "placed_x2",
    "placed_y2",
    "maximum_iou",
    "maximum_existing_occlusion",
    "placement_retries",
    "placement_seed",
    "source_foreground_mean_luminance",
    "source_mask_largest_component_fraction",
    "source_foreground_bbox_fill",
]

BACKGROUND_FIELDS = [
    "synthetic_image_path",
    "image_path",
    "content_hash",
    "source_group_id",
    "base_label_path",
    "retained_object_count",
    "per_image_seed",
]

FAILED_FIELDS = [
    "sample_index",
    "sample_attempt",
    "base_image_path",
    "attempt_seed",
    "reason",
]


def _git_revision(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _load_base_labels(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid base label at line {line_number}: {label_path}")
        class_id = int(values[0])
        normalized = tuple(float(value) for value in values[1:])
        box = yolo_to_xyxy(*normalized, width, height)
        rows.append({"class_id": class_id, "normalized": normalized, "box": box})
    if not rows:
        raise ValueError(f"Base canvas has no labels: {label_path}")
    return rows


def _actual_bank_path(
    canonical: str, logical_output_root: Path, physical_output_root: Path
) -> Path:
    relative = Path(canonical).relative_to(logical_output_root)
    return physical_output_root / relative


def _source_quality_metrics(
    row: dict[str, str],
    logical_output_root: Path,
    bank_physical_output_root: Path,
) -> dict[str, float]:
    crop_path = _actual_bank_path(
        row["crop_output_path"], logical_output_root, bank_physical_output_root
    )
    with Image.open(crop_path) as source:
        rgba = np.asarray(source.convert("RGBA"))
    foreground = rgba[:, :, 3] > 16
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count == 0:
        return {
            "foreground_mean_luminance": 0.0,
            "largest_component_fraction": 0.0,
            "foreground_bbox_fill": 0.0,
        }
    rgb = rgba[:, :, :3].astype(np.float32)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    component_count, labels = cv2.connectedComponents(foreground.astype(np.uint8), 8)
    component_sizes = [
        int(np.count_nonzero(labels == index)) for index in range(1, component_count)
    ]
    largest_component_fraction = max(component_sizes, default=0) / foreground_count
    mask_box = foreground_mask_bbox(foreground.astype(np.uint8) * 255)
    bbox_area = (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])
    return {
        "foreground_mean_luminance": float(luminance[foreground].mean()),
        "largest_component_fraction": largest_component_fraction,
        "foreground_bbox_fill": foreground_count / bbox_area,
    }


def _transform_object(
    row: dict[str, str],
    config: SyntheticConfig,
    rng: np.random.Generator,
    logical_output_root: Path,
    bank_physical_output_root: Path,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    crop_path = _actual_bank_path(
        row["crop_output_path"], logical_output_root, bank_physical_output_root
    )
    with Image.open(crop_path) as source:
        rgba = source.convert("RGBA")
    rgb = rgba.convert("RGB")
    alpha = rgba.getchannel("A")
    transforms = config.transforms
    scale = float(rng.uniform(*map(float, transforms["scale"])))
    target_width = max(1, round(rgb.width * scale))
    target_height = max(1, round(rgb.height * scale))
    rgb = rgb.resize((target_width, target_height), Image.Resampling.BICUBIC)
    alpha = alpha.resize((target_width, target_height), Image.Resampling.BILINEAR)
    horizontal_flip = bool(rng.random() < float(transforms["horizontal_flip_probability"]))
    if horizontal_flip:
        rgb = ImageOps.mirror(rgb)
        alpha = ImageOps.mirror(alpha)
    rotation = float(rng.uniform(*map(float, transforms["rotation_degrees"])))
    rgb = rgb.rotate(rotation, Image.Resampling.BICUBIC, expand=True)
    alpha = alpha.rotate(rotation, Image.Resampling.BILINEAR, expand=True)
    brightness = float(rng.uniform(*map(float, transforms["brightness"])))
    contrast = float(rng.uniform(*map(float, transforms["contrast"])))
    saturation = float(rng.uniform(*map(float, transforms["saturation"])))
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    blur_radius = 0.0
    if rng.random() < float(transforms["blur_probability"]):
        blur_radius = float(rng.uniform(*map(float, transforms["blur_radius"])))
        rgb = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
    noise_std = 0.0
    if rng.random() < float(transforms["noise_probability"]):
        noise_std = float(rng.uniform(*map(float, transforms["noise_standard_deviation"])))
        array = np.asarray(rgb).astype(np.float32)
        array += rng.normal(0, noise_std, array.shape)
        rgb = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    feather_radius = float(rng.uniform(*map(float, transforms["alpha_feather_radius"])))
    alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))
    alpha_array = np.asarray(alpha)
    bbox = foreground_mask_bbox(alpha_array, int(config.placement["foreground_alpha_threshold"]))
    rgb = rgb.crop(bbox)
    alpha_array = alpha_array[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    return (
        rgb,
        alpha_array,
        {
            "scale": scale,
            "rotation_degrees": rotation,
            "horizontal_flip": horizontal_flip,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "blur_radius": blur_radius,
            "noise_standard_deviation": noise_std,
            "alpha_feather_radius": feather_radius,
        },
    )


def _place_object(
    canvas: Image.Image,
    rgb: Image.Image,
    alpha: np.ndarray,
    existing_boxes: list[tuple[int, int, int, int]],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], float, float, int] | None:
    width, height = rgb.size
    minimum = int(config.placement["minimum_dimension_pixels"])
    maximum_fraction = float(config.placement["maximum_image_fraction"])
    if width < minimum or height < minimum:
        return None
    if width > canvas.width * maximum_fraction or height > canvas.height * maximum_fraction:
        return None
    if width > canvas.width or height > canvas.height:
        return None
    retries = int(config.placement["maximum_retries_per_object"])
    maximum_iou_allowed = float(config.placement["maximum_iou"])
    maximum_occlusion_allowed = float(config.placement["maximum_existing_object_occlusion"])
    local_bbox = foreground_mask_bbox(alpha, int(config.placement["foreground_alpha_threshold"]))
    for retry in range(retries):
        origin_x = int(rng.integers(0, canvas.width - width + 1))
        origin_y = int(rng.integers(0, canvas.height - height + 1))
        candidate = (
            origin_x + local_bbox[0],
            origin_y + local_bbox[1],
            origin_x + local_bbox[2],
            origin_y + local_bbox[3],
        )
        ious = [box_iou(candidate, existing) for existing in existing_boxes]
        occlusions = [existing_object_occlusion(candidate, existing) for existing in existing_boxes]
        maximum_iou = max(ious, default=0.0)
        maximum_occlusion = max(occlusions, default=0.0)
        if maximum_iou > maximum_iou_allowed or maximum_occlusion > maximum_occlusion_allowed:
            continue
        canvas.paste(rgb, (origin_x, origin_y), Image.fromarray(alpha))
        return candidate, maximum_iou, maximum_occlusion, retry
    return None


def _format_label(class_id: int, normalized: tuple[float, float, float, float]) -> str:
    return f"{class_id} " + " ".join(f"{value:.10f}" for value in normalized)


def _class_targets(
    train_rows: list[dict[str, str]], project_root: Path, class_count: int
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in train_rows:
        for line in (project_root / row["label_path"]).read_text(encoding="utf-8").splitlines():
            class_id = int(line.split()[0])
            if not 0 <= class_id < class_count:
                raise ValueError(f"Unsupported train class ID: {class_id}")
            counts[class_id] += 1
    if set(counts) != set(range(class_count)):
        raise ValueError("Real-train labels do not cover every configured class")
    return counts


def generate_synthetic_pool(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    manifest_dir: Path,
    bank_manifest_dir: Path,
    bank_physical_output_root: Path,
    count: int,
    ensure_class_coverage: bool = False,
) -> dict[str, Any]:
    """Generate a new pool and provenance manifests without overwriting any output."""

    images_dir = physical_output_root / "images"
    labels_dir = physical_output_root / "labels"
    if images_dir.exists() or labels_dir.exists():
        raise FileExistsError("Refusing to overwrite existing synthetic images or labels")
    for name in (
        "synthetic_images.csv",
        "synthetic_sources.csv",
        "synthetic_backgrounds.csv",
        "failed_generation_attempts.csv",
        "generation_metadata.json",
    ):
        if (manifest_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing synthetic manifest: {name}")
    split_dir = project_root / Path(config.dataset["active_split_directory"])
    train_rows = read_csv(split_dir / "real_train.csv")
    included_bank_rows = [
        row
        for row in read_csv(bank_manifest_dir / "object_bank.csv")
        if row["inclusion_status"] == "included"
    ]
    allowed_methods = set(config.sampling["allowed_extraction_methods"])
    minimum_coverage = float(config.sampling["minimum_generation_mask_coverage"])
    maximum_coverage = float(config.sampling["maximum_generation_mask_coverage"])
    coverage_filtered_rows = [
        row
        for row in included_bank_rows
        if row["extraction_method"] in allowed_methods
        and minimum_coverage <= float(row["mask_coverage"]) <= maximum_coverage
    ]
    minimum_luminance = float(config.sampling["minimum_foreground_mean_luminance"])
    minimum_component = float(config.sampling["minimum_largest_component_fraction"])
    minimum_bbox_fill = float(config.sampling["minimum_foreground_bbox_fill"])
    source_quality = {
        row["object_id"]: _source_quality_metrics(
            row,
            Path(config.dataset["output_directory"]),
            bank_physical_output_root,
        )
        for row in coverage_filtered_rows
    }
    bank_rows = [
        row
        for row in coverage_filtered_rows
        if source_quality[row["object_id"]]["foreground_mean_luminance"] >= minimum_luminance
        and source_quality[row["object_id"]]["largest_component_fraction"] >= minimum_component
        and source_quality[row["object_id"]]["foreground_bbox_fill"] >= minimum_bbox_fill
    ]
    object_bank_identity = calculate_object_bank_identity(bank_manifest_dir)
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in bank_rows:
        by_class[int(row["class_id"])].append(row)
    missing_eligible_classes = [
        config.class_names[index] for index in range(len(config.class_names)) if not by_class[index]
    ]
    if missing_eligible_classes:
        raise ValueError(
            "Generation-quality filtering removed every source for classes: "
            + ", ".join(missing_eligible_classes)
        )
    class_counts = _class_targets(train_rows, project_root, len(config.class_names))
    total_real_objects = sum(class_counts.values())
    probabilities = np.array(
        [class_counts[index] / total_real_objects for index in range(len(config.class_names))]
    )
    base_rows = sorted(train_rows, key=lambda row: row["image_path"])
    base_rng = random.Random(derive_seed(config.root_seed, "base-order", config.split_identity))
    base_rng.shuffle(base_rows)
    if count > len(base_rows):
        raise ValueError(
            "Pool size cannot exceed train canvases without an explicit replacement policy"
        )

    synthetic_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    rejected_placement_attempts = 0
    for sample_index in range(count):
        base_row = base_rows[sample_index]
        accepted = False
        for sample_attempt in range(int(config.placement["maximum_sample_attempts"])):
            sample_seed = derive_seed(
                config.root_seed,
                "sample",
                sample_index,
                sample_attempt,
                config.configuration_hash,
                config.split_identity,
                object_bank_identity,
            )
            rng = np.random.default_rng(sample_seed)
            base_path = project_root / base_row["image_path"]
            with Image.open(base_path) as base_source:
                canvas = base_source.convert("RGB")
            base_labels = _load_base_labels(
                project_root / base_row["label_path"], canvas.width, canvas.height
            )
            existing_boxes = [row["box"] for row in base_labels]
            pasted_records: list[dict[str, Any]] = []
            pasted_labels: list[tuple[int, tuple[float, float, float, float]]] = []
            requested_pastes = int(
                rng.integers(
                    int(config.sampling["pasted_objects_per_image"]["minimum"]),
                    int(config.sampling["pasted_objects_per_image"]["maximum"]) + 1,
                )
            )
            for placement_index in range(requested_pastes):
                if (
                    ensure_class_coverage
                    and sample_index < len(config.class_names)
                    and placement_index == 0
                ):
                    class_id = sample_index
                else:
                    class_id = int(rng.choice(len(config.class_names), p=probabilities))
                candidates = by_class[class_id]
                different_group = [
                    row
                    for row in candidates
                    if row["source_group_id"] != base_row["source_group_id"]
                ]
                if config.sampling["prefer_different_source_group"] and different_group:
                    candidates = different_group
                placement_seed = derive_seed(sample_seed, "placement", placement_index)
                placement_rng = np.random.default_rng(placement_seed)
                object_row = candidates[int(placement_rng.integers(0, len(candidates)))]
                try:
                    rgb, alpha, transformation = _transform_object(
                        object_row,
                        config,
                        placement_rng,
                        Path(config.dataset["output_directory"]),
                        bank_physical_output_root,
                    )
                    placed = _place_object(
                        canvas, rgb, alpha, existing_boxes, config, placement_rng
                    )
                except (OSError, ValueError) as error:
                    failed_rows.append(
                        {
                            "sample_index": sample_index,
                            "sample_attempt": sample_attempt,
                            "base_image_path": base_row["image_path"],
                            "attempt_seed": sample_seed,
                            "reason": f"object_transform_failed:{error}",
                        }
                    )
                    placed = None
                if placed is None:
                    rejected_placement_attempts += int(
                        config.placement["maximum_retries_per_object"]
                    )
                    continue
                candidate_box, maximum_iou, maximum_occlusion, retries = placed
                rejected_placement_attempts += retries
                existing_boxes.append(candidate_box)
                normalized = xyxy_to_yolo(candidate_box, canvas.width, canvas.height)
                pasted_labels.append((class_id, normalized))
                pasted_records.append(
                    {
                        "object": object_row,
                        "class_id": class_id,
                        "transformation": transformation,
                        "candidate_box": candidate_box,
                        "maximum_iou": maximum_iou,
                        "maximum_occlusion": maximum_occlusion,
                        "retries": retries,
                        "placement_seed": placement_seed,
                    }
                )
            if not pasted_records:
                failed_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_attempt": sample_attempt,
                        "base_image_path": base_row["image_path"],
                        "attempt_seed": sample_seed,
                        "reason": "no_valid_paste_after_retries",
                    }
                )
                continue
            filename_digest = stable_json_hash(
                [config.split_identity, object_bank_identity, sample_index, sample_seed]
            )[:12]
            filename = f"synthetic-{sample_index:06d}-{filename_digest}"
            image_actual = images_dir / f"{filename}.jpg"
            label_actual = labels_dir / f"{filename}.txt"
            if image_actual.exists() or label_actual.exists():
                raise FileExistsError(f"Synthetic filename collision: {filename}")
            image_actual.parent.mkdir(parents=True, exist_ok=True)
            label_actual.parent.mkdir(parents=True, exist_ok=True)
            jpeg_quality = int(
                rng.integers(
                    int(config.transforms["jpeg_quality"][0]),
                    int(config.transforms["jpeg_quality"][1]) + 1,
                )
            )
            canvas.save(
                image_actual,
                "JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
            label_lines = [
                _format_label(row["class_id"], row["normalized"]) for row in base_labels
            ] + [_format_label(class_id, normalized) for class_id, normalized in pasted_labels]
            label_actual.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            canonical_image = (logical_output_root / "images" / image_actual.name).as_posix()
            canonical_label = (logical_output_root / "labels" / label_actual.name).as_posix()
            class_ids = sorted(
                {row["class_id"] for row in base_labels} | {row[0] for row in pasted_labels}
            )
            synthetic_rows.append(
                {
                    "synthetic_image_path": canonical_image,
                    "synthetic_label_path": canonical_label,
                    "output_image_hash": sha256_file(image_actual),
                    "output_label_hash": sha256_file(label_actual),
                    "base_image_path": base_row["image_path"],
                    "base_image_hash": base_row["content_hash"],
                    "base_source_group_id": base_row["source_group_id"],
                    "pasted_object_count": len(pasted_records),
                    "retained_base_object_count": len(base_labels),
                    "class_ids_present": ";".join(map(str, class_ids)),
                    "total_object_count": len(label_lines),
                    "image_width": canvas.width,
                    "image_height": canvas.height,
                    "per_image_seed": sample_seed,
                    "generator_configuration_hash": config.configuration_hash,
                    "object_bank_identity": object_bank_identity,
                    "active_real_split_identity": config.split_identity,
                    "jpeg_quality": jpeg_quality,
                    "generation_status": "accepted",
                }
            )
            background_rows.append(
                {
                    "synthetic_image_path": canonical_image,
                    "image_path": base_row["image_path"],
                    "content_hash": base_row["content_hash"],
                    "source_group_id": base_row["source_group_id"],
                    "base_label_path": base_row["label_path"],
                    "retained_object_count": len(base_labels),
                    "per_image_seed": sample_seed,
                }
            )
            for record in pasted_records:
                object_row = record["object"]
                transformation = record["transformation"]
                x1, y1, x2, y2 = record["candidate_box"]
                source_rows.append(
                    {
                        "synthetic_image_path": canonical_image,
                        "object_id": object_row["object_id"],
                        "class_id": record["class_id"],
                        "class_name": config.class_names[record["class_id"]],
                        "image_path": object_row["image_path"],
                        "content_hash": object_row["content_hash"],
                        "source_group_id": object_row["source_group_id"],
                        "duplicate_group_id": object_row["duplicate_group_id"],
                        "scale": f"{transformation['scale']:.10f}",
                        "rotation_degrees": f"{transformation['rotation_degrees']:.10f}",
                        "horizontal_flip": str(transformation["horizontal_flip"]).lower(),
                        "brightness": f"{transformation['brightness']:.10f}",
                        "contrast": f"{transformation['contrast']:.10f}",
                        "saturation": f"{transformation['saturation']:.10f}",
                        "blur_radius": f"{transformation['blur_radius']:.10f}",
                        "noise_standard_deviation": (
                            f"{transformation['noise_standard_deviation']:.10f}"
                        ),
                        "alpha_feather_radius": f"{transformation['alpha_feather_radius']:.10f}",
                        "placed_x1": x1,
                        "placed_y1": y1,
                        "placed_x2": x2,
                        "placed_y2": y2,
                        "maximum_iou": f"{record['maximum_iou']:.10f}",
                        "maximum_existing_occlusion": f"{record['maximum_occlusion']:.10f}",
                        "placement_retries": record["retries"],
                        "placement_seed": record["placement_seed"],
                        "source_foreground_mean_luminance": (
                            f"{source_quality[object_row['object_id']]['foreground_mean_luminance']:.10f}"
                        ),
                        "source_mask_largest_component_fraction": (
                            f"{source_quality[object_row['object_id']]['largest_component_fraction']:.10f}"
                        ),
                        "source_foreground_bbox_fill": (
                            f"{source_quality[object_row['object_id']]['foreground_bbox_fill']:.10f}"
                        ),
                    }
                )
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"Unable to generate accepted synthetic sample {sample_index} after all attempts"
            )

    write_csv(manifest_dir / "synthetic_images.csv", SYNTHETIC_IMAGE_FIELDS, synthetic_rows)
    write_csv(manifest_dir / "synthetic_sources.csv", SOURCE_FIELDS, source_rows)
    write_csv(manifest_dir / "synthetic_backgrounds.csv", BACKGROUND_FIELDS, background_rows)
    write_csv(manifest_dir / "failed_generation_attempts.csv", FAILED_FIELDS, failed_rows)
    data_yaml = {
        "path": ".",
        "train": "images",
        "val": None,
        "names": {index: name for index, name in enumerate(config.class_names)},
        "synthetic_train_only": True,
        "active_real_validation_is_materialized_separately": True,
    }
    (physical_output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    manifest_hashes = {
        name: sha256_file(manifest_dir / name)
        for name in (
            "synthetic_images.csv",
            "synthetic_sources.csv",
            "synthetic_backgrounds.csv",
            "object_bank.csv",
            "excluded_objects.csv",
            "failed_generation_attempts.csv",
        )
        if (manifest_dir / name).is_file()
    }
    # In reproduction mode bank manifests live in the frozen bank directory.
    for name in ("object_bank.csv", "excluded_objects.csv"):
        if name not in manifest_hashes:
            manifest_hashes[name] = sha256_file(bank_manifest_dir / name)
    manifest_hashes["materialized_data.yaml"] = sha256_file(physical_output_root / "data.yaml")
    combined_identity = stable_json_hash(manifest_hashes)
    actual_pasted = Counter(int(row["class_id"]) for row in source_rows)
    transformation_usage = {
        "horizontal_flip": sum(row["horizontal_flip"] == "true" for row in source_rows),
        "blur": sum(float(row["blur_radius"]) > 0 for row in source_rows),
        "noise": sum(float(row["noise_standard_deviation"]) > 0 for row in source_rows),
    }
    metadata = {
        "status": "frozen" if count == int(config.dataset["full_pool_size"]) else "smoke",
        "active_split_identity": config.split_identity,
        "code_revision": _git_revision(project_root),
        "root_seed": config.root_seed,
        "generator_version": config.dataset["generator_version"],
        "generation_mode": config.dataset["mode"],
        "class_coverage_preview": ensure_class_coverage,
        "configuration_hash": config.configuration_hash,
        "object_bank_identity": object_bank_identity,
        "object_bank_generation_filter": {
            "included_bank_objects": len(included_bank_rows),
            "eligible_generation_objects": len(bank_rows),
            "allowed_extraction_methods": sorted(allowed_methods),
            "minimum_mask_coverage": minimum_coverage,
            "maximum_mask_coverage": maximum_coverage,
            "minimum_foreground_mean_luminance": minimum_luminance,
            "minimum_largest_component_fraction": minimum_component,
            "minimum_foreground_bbox_fill": minimum_bbox_fill,
            "coverage_and_method_eligible_objects": len(coverage_filtered_rows),
            "eligible_class_counts": {
                config.class_names[index]: len(by_class[index])
                for index in range(len(config.class_names))
            },
        },
        "generated_count": len(synthetic_rows),
        "pasted_object_count": len(source_rows),
        "failed_sample_attempt_count": len(failed_rows),
        "rejected_placement_attempt_count": rejected_placement_attempts,
        "started_and_completed_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_excluded_from_pool_identity": True,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pillow": Image.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "real_train_class_counts": {
            config.class_names[index]: class_counts[index]
            for index in range(len(config.class_names))
        },
        "real_train_class_proportions": {
            config.class_names[index]: class_counts[index] / total_real_objects
            for index in range(len(config.class_names))
        },
        "target_pasted_class_proportions": {
            config.class_names[index]: float(probabilities[index])
            for index in range(len(config.class_names))
        },
        "actual_pasted_class_counts": {
            config.class_names[index]: actual_pasted[index]
            for index in range(len(config.class_names))
        },
        "actual_pasted_class_proportions": {
            config.class_names[index]: actual_pasted[index] / max(len(source_rows), 1)
            for index in range(len(config.class_names))
        },
        "transformation_usage": transformation_usage,
        "output_manifest_hashes": manifest_hashes,
        "combined_synthetic_pool_identity": combined_identity,
        "known_nondeterminism": (
            "Timestamps are observational and excluded from identity. Pixel reproduction is "
            "verified in the recorded Pillow/OpenCV/NumPy environment; library codec changes may "
            "change JPEG bytes across environments."
        ),
    }
    (manifest_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_synthetic_reproduction(
    config: SyntheticConfig,
    project_root: Path,
    frozen_output_root: Path,
    frozen_manifest_dir: Path,
) -> dict[str, Any]:
    frozen_metadata = json.loads(
        (frozen_manifest_dir / "generation_metadata.json").read_text(encoding="utf-8")
    )
    count = int(frozen_metadata["generated_count"])
    with tempfile.TemporaryDirectory(prefix="synthdet-synthetic-reproduction-") as temporary:
        temp_root = Path(temporary)
        temp_output = temp_root / "output"
        temp_manifests = temp_root / "manifests"
        reproduced = generate_synthetic_pool(
            config,
            project_root,
            temp_output,
            Path(config.dataset["output_directory"]),
            temp_manifests,
            frozen_manifest_dir,
            frozen_output_root,
            count,
            ensure_class_coverage=bool(frozen_metadata.get("class_coverage_preview", False)),
        )
        if reproduced["combined_synthetic_pool_identity"] != frozen_metadata.get(
            "combined_synthetic_pool_identity"
        ):
            raise ValueError(
                "Synthetic pool identity reproduction mismatch: "
                f"expected {frozen_metadata.get('combined_synthetic_pool_identity')}, "
                f"got {reproduced['combined_synthetic_pool_identity']}"
            )
        for row in read_csv(temp_manifests / "synthetic_images.csv"):
            relative_image = Path(row["synthetic_image_path"]).relative_to(
                Path(config.dataset["output_directory"])
            )
            relative_label = Path(row["synthetic_label_path"]).relative_to(
                Path(config.dataset["output_directory"])
            )
            if sha256_file(temp_output / relative_image) != row["output_image_hash"]:
                raise ValueError("Reproduced synthetic image hash mismatch")
            if sha256_file(temp_output / relative_label) != row["output_label_hash"]:
                raise ValueError("Reproduced synthetic label hash mismatch")
    return reproduced
