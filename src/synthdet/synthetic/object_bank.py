"""Train-only Aquarium object-bank extraction with GrabCut and documented fallback."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from synthdet.synthetic.contracts import (
    SyntheticConfig,
    derive_seed,
    read_csv,
    sha256_file,
    stable_json_hash,
    write_csv,
)
from synthdet.synthetic.geometry import yolo_to_xyxy

OBJECT_BANK_FIELDS = [
    "object_id",
    "class_id",
    "class_name",
    "image_path",
    "label_path",
    "content_hash",
    "source_group_id",
    "duplicate_group_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "yolo_x_center",
    "yolo_y_center",
    "yolo_width",
    "yolo_height",
    "image_width",
    "image_height",
    "crop_width",
    "crop_height",
    "crop_area",
    "object_size_category",
    "touches_left_edge",
    "touches_top_edge",
    "touches_right_edge",
    "touches_bottom_edge",
    "extraction_method",
    "crop_output_path",
    "mask_output_path",
    "crop_hash",
    "mask_hash",
    "mask_coverage",
    "foreground_area",
    "extraction_quality_status",
    "extraction_quality_notes",
    "inclusion_status",
    "exclusion_reason",
]


def _size_category(area: int) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def _fallback_mask(width: int, height: int, feather: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    distance = np.minimum.reduce([xx, width - 1 - xx, yy, height - 1 - yy]).astype(np.float32)
    if feather <= 0:
        return np.full((height, width), 255, dtype=np.uint8)
    return np.clip((distance + 1) / feather, 0, 1).astype(np.float32) * 255


def _extract_mask(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    config: SyntheticConfig,
    object_seed: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2].copy()
    width, height = x2 - x1, y2 - y1
    context_fraction = float(config.object_bank["context_fraction"])
    pad_x = max(2, round(width * context_fraction))
    pad_y = max(2, round(height * context_fraction))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(rgb.shape[1], x2 + pad_x), min(rgb.shape[0], y2 + pad_y)
    context = rgb[cy1:cy2, cx1:cx2]
    mask = np.zeros(context.shape[:2], dtype=np.uint8)
    rect = (x1 - cx1, y1 - cy1, width, height)
    method = "grabcut"
    note = "bbox-initialized GrabCut; mask is an estimate, not segmentation ground truth"
    try:
        cv2.setRNGSeed(int(object_seed % (2**31 - 1)))
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            cv2.cvtColor(context, cv2.COLOR_RGB2BGR),
            mask,
            rect,
            background,
            foreground,
            int(config.object_bank["grabcut_iterations"]),
            cv2.GC_INIT_WITH_RECT,
        )
        alpha_context = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
            np.uint8
        )
        alpha = alpha_context[y1 - cy1 : y2 - cy1, x1 - cx1 : x2 - cx1]
        coverage = float(np.count_nonzero(alpha)) / alpha.size
        if not (
            float(config.object_bank["minimum_mask_coverage"])
            <= coverage
            <= float(config.object_bank["maximum_mask_coverage"])
        ):
            raise ValueError(f"degenerate GrabCut coverage {coverage:.6f}")
    except (cv2.error, ValueError) as error:
        alpha = _fallback_mask(
            width, height, int(config.object_bank["fallback_feather_pixels"])
        ).astype(np.uint8)
        method = "feathered_rectangle_fallback"
        note = f"GrabCut unavailable/degenerate ({error}); softly feathered rectangle used"
    return crop, alpha, method, note


def _write_contact_sheets(
    included: list[dict[str, str]],
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    output_dir: Path,
    examples_per_class: int,
) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in included:
        grouped[row["class_name"]].append(row)
    outputs: list[str] = []
    tile_width, panel_height, label_height, columns = 600, 180, 44, 2
    for class_name, rows in sorted(grouped.items()):
        if len(rows) > examples_per_class:
            indices = np.linspace(0, len(rows) - 1, examples_per_class, dtype=int)
            selected = [rows[index] for index in indices]
        else:
            selected = rows
        page_size = 8
        for offset in range(0, len(selected), page_size):
            page = selected[offset : offset + page_size]
            sheet_rows = math.ceil(len(page) / columns)
            canvas = Image.new(
                "RGB", (columns * tile_width, sheet_rows * (panel_height + label_height)), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(page):
                with Image.open(project_root / row["image_path"]) as source:
                    context = source.convert("RGB")
                    context_draw = ImageDraw.Draw(context)
                    box = tuple(
                        int(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                    )
                    context_draw.rectangle(box, outline="red", width=max(2, context.width // 300))
                    context = ImageOps.contain(context, (190, panel_height - 8))
                crop_actual = physical_output_root / Path(row["crop_output_path"]).relative_to(
                    logical_output_root
                )
                mask_actual = physical_output_root / Path(row["mask_output_path"]).relative_to(
                    logical_output_root
                )
                with Image.open(crop_actual) as crop_image:
                    crop = ImageOps.contain(crop_image.convert("RGBA"), (190, panel_height - 8))
                with Image.open(mask_actual) as mask_image:
                    mask = ImageOps.contain(
                        mask_image.convert("L"), (190, panel_height - 8)
                    ).convert("RGB")
                column, sheet_row = index % columns, index // columns
                origin_x = column * tile_width
                origin_y = sheet_row * (panel_height + label_height)
                for panel_index, panel in enumerate((context, crop.convert("RGB"), mask)):
                    panel_x = origin_x + panel_index * 200 + (190 - panel.width) // 2
                    panel_y = origin_y + (panel_height - panel.height) // 2
                    canvas.paste(panel, (panel_x, panel_y))
                draw.text(
                    (origin_x + 5, origin_y + panel_height + 2),
                    f"{row['object_id']} | {class_name} | {row['extraction_method']}",
                    fill="black",
                )
                draw.text(
                    (origin_x + 5, origin_y + panel_height + 22),
                    f"coverage={float(row['mask_coverage']):.3f} | context / crop / mask",
                    fill="#555555",
                )
            target = output_dir / f"object-bank-{class_name}-page-{offset // page_size + 1:02d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target, quality=92)
            outputs.append(target.as_posix())
    return outputs


def build_object_bank(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    """Extract every eligible train object and retain explicit exclusion records."""

    bank_root = physical_output_root / "object_bank"
    masks_root = physical_output_root / "masks"
    if bank_root.exists() or masks_root.exists():
        raise FileExistsError("Refusing to overwrite an existing object bank or mask directory")
    if (manifest_dir / "object_bank.csv").exists() or (
        manifest_dir / "excluded_objects.csv"
    ).exists():
        raise FileExistsError("Refusing to overwrite existing object-bank manifests")
    train_manifest = Path(config.dataset["active_split_directory"]) / "real_train.csv"
    train_rows = read_csv(project_root / train_manifest)
    duplicate_rows = read_csv(
        project_root / Path(config.dataset["active_split_directory"]) / "duplicate_groups.csv"
    )
    duplicate_by_path = {
        row["image_path"]: row.get("duplicate_group_id", "") for row in duplicate_rows
    }
    all_rows: list[dict[str, Any]] = []
    for image_index, manifest_row in enumerate(
        sorted(train_rows, key=lambda row: row["image_path"])
    ):
        image_path = project_root / manifest_row["image_path"]
        label_path = project_root / manifest_row["label_path"]
        try:
            with Image.open(image_path) as source:
                rgb = np.asarray(source.convert("RGB"))
        except (OSError, ValueError) as error:
            all_rows.append(
                {
                    **{field: "" for field in OBJECT_BANK_FIELDS},
                    "object_id": f"failed-image-{image_index:04d}",
                    "image_path": manifest_row["image_path"],
                    "label_path": manifest_row["label_path"],
                    "content_hash": manifest_row["content_hash"],
                    "source_group_id": manifest_row["source_group_id"],
                    "inclusion_status": "excluded",
                    "exclusion_reason": f"failed_image_read:{error}",
                    "extraction_quality_status": "failed",
                }
            )
            continue
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line_index, line in enumerate(lines, start=1):
            base: dict[str, Any] = {field: "" for field in OBJECT_BANK_FIELDS}
            object_digest = stable_json_hash(
                [manifest_row["image_path"], line_index, line.strip()]
            )[:16]
            base.update(
                {
                    "object_id": f"obj-{object_digest}",
                    "image_path": manifest_row["image_path"],
                    "label_path": manifest_row["label_path"],
                    "content_hash": manifest_row["content_hash"],
                    "source_group_id": manifest_row["source_group_id"],
                    "duplicate_group_id": duplicate_by_path.get(manifest_row["image_path"], ""),
                }
            )
            try:
                values = line.split()
                if len(values) != 5:
                    raise ValueError("invalid_box_field_count")
                class_id = int(values[0])
                if not 0 <= class_id < len(config.class_names):
                    raise ValueError("unsupported_class_id")
                normalized = tuple(float(value) for value in values[1:])
                box = yolo_to_xyxy(*normalized, rgb.shape[1], rgb.shape[0])
                x1, y1, x2, y2 = box
                crop_width, crop_height = x2 - x1, y2 - y1
                crop_area = crop_width * crop_height
                if crop_width < int(
                    config.object_bank["minimum_crop_dimension"]
                ) or crop_height < int(config.object_bank["minimum_crop_dimension"]):
                    raise ValueError("extremely_small_crop_dimension")
                if crop_area < int(config.object_bank["minimum_crop_area"]):
                    raise ValueError("extremely_small_crop_area")
                crop, alpha, method, note = _extract_mask(
                    rgb,
                    box,
                    config,
                    derive_seed(config.root_seed, "object", base["object_id"]),
                )
                if crop.size == 0:
                    raise ValueError("empty_crop")
                foreground_area = int(np.count_nonzero(alpha > 16))
                if foreground_area <= 0:
                    raise ValueError("failed_mask_empty")
                mask_coverage = foreground_area / alpha.size
                crop_relative = (
                    logical_output_root / "object_bank" / "crops" / f"{base['object_id']}.png"
                )
                mask_relative = logical_output_root / "masks" / f"{base['object_id']}.png"
                crop_actual = bank_root / "crops" / f"{base['object_id']}.png"
                mask_actual = masks_root / f"{base['object_id']}.png"
                crop_actual.parent.mkdir(parents=True, exist_ok=True)
                mask_actual.parent.mkdir(parents=True, exist_ok=True)
                rgba = np.dstack((crop, alpha))
                Image.fromarray(rgba).save(crop_actual)
                Image.fromarray(alpha).save(mask_actual)
                base.update(
                    {
                        "class_id": class_id,
                        "class_name": config.class_names[class_id],
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "yolo_x_center": f"{normalized[0]:.10f}",
                        "yolo_y_center": f"{normalized[1]:.10f}",
                        "yolo_width": f"{normalized[2]:.10f}",
                        "yolo_height": f"{normalized[3]:.10f}",
                        "image_width": rgb.shape[1],
                        "image_height": rgb.shape[0],
                        "crop_width": crop_width,
                        "crop_height": crop_height,
                        "crop_area": crop_area,
                        "object_size_category": _size_category(crop_area),
                        "touches_left_edge": str(x1 == 0).lower(),
                        "touches_top_edge": str(y1 == 0).lower(),
                        "touches_right_edge": str(x2 == rgb.shape[1]).lower(),
                        "touches_bottom_edge": str(y2 == rgb.shape[0]).lower(),
                        "extraction_method": method,
                        "crop_output_path": crop_relative.as_posix(),
                        "mask_output_path": mask_relative.as_posix(),
                        "crop_hash": sha256_file(crop_actual),
                        "mask_hash": sha256_file(mask_actual),
                        "mask_coverage": f"{mask_coverage:.10f}",
                        "foreground_area": foreground_area,
                        "extraction_quality_status": "usable_estimate",
                        "extraction_quality_notes": note,
                        "inclusion_status": "included",
                        "exclusion_reason": "",
                    }
                )
            except (OSError, TypeError, ValueError) as error:
                base.update(
                    {
                        "inclusion_status": "excluded",
                        "exclusion_reason": str(error),
                        "extraction_quality_status": "failed",
                    }
                )
            all_rows.append(base)
    all_rows.sort(key=lambda row: row["object_id"])
    included = [row for row in all_rows if row["inclusion_status"] == "included"]
    excluded = [row for row in all_rows if row["inclusion_status"] == "excluded"]
    if not included:
        raise ValueError("Object bank contains no usable objects")
    if set(row["class_name"] for row in included) != set(config.class_names):
        raise ValueError("Object bank does not cover every configured class")
    object_bank_path = manifest_dir / "object_bank.csv"
    excluded_path = manifest_dir / "excluded_objects.csv"
    write_csv(object_bank_path, OBJECT_BANK_FIELDS, all_rows)
    write_csv(excluded_path, OBJECT_BANK_FIELDS, excluded)
    bank_identity = stable_json_hash(
        {
            "object_bank_csv": sha256_file(object_bank_path),
            "excluded_objects_csv": sha256_file(excluded_path),
            "included_crop_hashes": [row["crop_hash"] for row in included],
            "included_mask_hashes": [row["mask_hash"] for row in included],
        }
    )
    preview_dir = physical_output_root / "previews" / "object_bank"
    contact_sheets = _write_contact_sheets(
        included,
        project_root,
        physical_output_root,
        logical_output_root,
        preview_dir,
        int(config.object_bank["contact_sheet_examples_per_class"]),
    )
    method_counts = Counter(row["extraction_method"] for row in included)
    summary = {
        "object_bank_identity": bank_identity,
        "total_records": len(all_rows),
        "included_objects": len(included),
        "excluded_objects": len(excluded),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in included).items())),
        "size_counts": dict(
            sorted(Counter(row["object_size_category"] for row in included).items())
        ),
        "extraction_method_counts": dict(sorted(method_counts.items())),
        "exclusion_reasons": dict(
            sorted(Counter(row["exclusion_reason"] for row in excluded).items())
        ),
        "mask_coverage": {
            "minimum": min(float(row["mask_coverage"]) for row in included),
            "mean": sum(float(row["mask_coverage"]) for row in included) / len(included),
            "maximum": max(float(row["mask_coverage"]) for row in included),
        },
        "contact_sheets": contact_sheets,
    }
    (physical_output_root / "previews" / "object_bank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def calculate_object_bank_identity(manifest_dir: Path) -> str:
    rows = read_csv(manifest_dir / "object_bank.csv")
    included = [row for row in rows if row["inclusion_status"] == "included"]
    return stable_json_hash(
        {
            "object_bank_csv": sha256_file(manifest_dir / "object_bank.csv"),
            "excluded_objects_csv": sha256_file(manifest_dir / "excluded_objects.csv"),
            "included_crop_hashes": [row["crop_hash"] for row in included],
            "included_mask_hashes": [row["mask_hash"] for row in included],
        }
    )
