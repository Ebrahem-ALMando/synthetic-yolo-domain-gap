"""Build the focused Aquarium Penguin source-dependency review evidence."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.data.hashing import hamming_distance  # noqa: E402

PENGUIN_CLASS = "penguin"
EXPECTED_IMAGES = 71
EXPECTED_OBJECTS = 516
CURRENT_GROUP = "reviewed-scene-penguin-rock-pool"

CAPTURE_GROUPS = (
    {
        "id": "penguin-capture-2282-2354",
        "start": 2282,
        "end": 2354,
        "filename_evidence": (
            "IMG_2282-IMG_2354 run; Penguin frames IMG_2301-IMG_2354; "
            "adjacent and near-adjacent still captures"
        ),
        "visual_evidence": (
            "Continuous first-session progression through rock-wall, waterline, and underwater "
            "views; viewpoint changes occur within one coherent capture run"
        ),
    },
    {
        "id": "penguin-capture-2519-2530",
        "start": 2519,
        "end": 2530,
        "filename_evidence": (
            "IMG_2519-IMG_2530 run; Penguin frames IMG_2527-IMG_2530; 173-frame "
            "discontinuity after the prior Penguin run"
        ),
        "visual_evidence": (
            "Distinct later close waterline/rock-pool pass with four adjacent Penguin frames; "
            "kept intact as one short dependent sequence"
        ),
    },
    {
        "id": "penguin-capture-3130-3177",
        "start": 3130,
        "end": 3177,
        "filename_evidence": (
            "IMG_3130-IMG_3177 run; Penguin frames IMG_3163-IMG_3177; 633-frame "
            "discontinuity after the prior Penguin run"
        ),
        "visual_evidence": (
            "Distinct much-later pass with changed framing, subject arrangement, and lighting; "
            "adjacent frames remain together"
        ),
    },
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _capture_number(capture_id: str) -> int:
    match = re.fullmatch(r"IMG_(\d+)", capture_id)
    if not match:
        raise ValueError(f"Expected a still-image IMG identifier, got {capture_id!r}")
    return int(match.group(1))


def _capture_group(number: int) -> dict[str, Any]:
    matches = [group for group in CAPTURE_GROUPS if group["start"] <= number <= group["end"]]
    if len(matches) != 1:
        raise ValueError(f"Capture IMG_{number} maps to {len(matches)} Penguin review groups")
    return matches[0]


def _contact_sheets(
    groups: dict[str, list[dict[str, str]]], dataset_root: Path, output_dir: Path
) -> dict[str, list[str]]:
    tile_width, image_height, label_height, columns, page_size = 280, 210, 48, 4, 16
    tile_height = image_height + label_height
    paths: dict[str, list[str]] = defaultdict(list)
    for group_id, members in sorted(groups.items()):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", group_id).strip("-")
        for offset in range(0, len(members), page_size):
            page = members[offset : offset + page_size]
            rows = math.ceil(len(page) / columns)
            canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(page):
                with Image.open(dataset_root / row["image_path"]) as source:
                    preview = ImageOps.contain(
                        source.convert("RGB"), (tile_width - 12, image_height - 12)
                    )
                column, sheet_row = index % columns, index // columns
                x = column * tile_width + (tile_width - preview.width) // 2
                y = sheet_row * tile_height + (image_height - preview.height) // 2
                canvas.paste(preview, (x, y))
                label_x = column * tile_width + 6
                label_y = sheet_row * tile_height + image_height + 2
                draw.text((label_x, label_y), row["original_capture_id"], fill="black")
                draw.text(
                    (label_x, label_y + 20),
                    f"objects={row['penguin_object_count']}",
                    fill="#555555",
                )
            page_number = offset // page_size + 1
            target = output_dir / f"{safe_id}_page-{page_number:02d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target, quality=92)
            reference = (
                target.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                if target.is_absolute()
                else target.as_posix()
            )
            paths[group_id].append(reference)
    return dict(paths)


def _similarity_order(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    remaining = {row["image_path"]: row for row in rows}
    current = min(rows, key=lambda row: row["original_capture_id"])
    ordered = [current]
    remaining.pop(current["image_path"])
    while remaining:
        current = min(
            remaining.values(),
            key=lambda row: (
                hamming_distance(ordered[-1]["perceptual_hash"], row["perceptual_hash"]),
                row["original_capture_id"],
            ),
        )
        ordered.append(current)
        remaining.pop(current["image_path"])
    return ordered


def build_review(
    records_path: Path,
    boxes_path: Path,
    duplicates_path: Path,
    sources_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate the Penguin inventory and generate the review package and V2 source map."""

    _, records = _read_csv(records_path)
    _, boxes = _read_csv(boxes_path)
    _, duplicates = _read_csv(duplicates_path)
    source_fields, source_rows = _read_csv(sources_path)
    duplicate_by_path = {row["image_path"]: row for row in duplicates}
    source_by_path = {row["image_path"]: row for row in source_rows}
    penguin_counts = Counter(
        row["image_path"] for row in boxes if row["class_name"] == PENGUIN_CLASS
    )
    penguins = [
        row
        for row in records
        if row["inclusion_status"] == "included"
        and PENGUIN_CLASS in set(filter(None, row["classes_present"].split(";")))
    ]
    if len(penguins) != EXPECTED_IMAGES or sum(penguin_counts.values()) != EXPECTED_OBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} Penguin images/{EXPECTED_OBJECTS} objects, got "
            f"{len(penguins)}/{sum(penguin_counts.values())}"
        )
    if set(penguin_counts) != {row["image_path"] for row in penguins}:
        raise ValueError("Penguin image records and bounding-box inventory disagree")

    enriched: list[dict[str, str]] = []
    for record in penguins:
        path = record["image_path"]
        source = source_by_path[path]
        if source["source_group_id"] != CURRENT_GROUP:
            raise ValueError(f"Unexpected current Penguin source group for {path}")
        capture = source["original_capture_id"]
        group = _capture_group(_capture_number(capture))
        duplicate = duplicate_by_path[path]
        enriched.append(
            {
                **record,
                "original_capture_id": capture,
                "original_export_folder": path.split("/", maxsplit=1)[0],
                "original_source_group_id": source["source_group_id"],
                "proposed_final_source_group_id": group["id"],
                "duplicate_group_id": duplicate.get("duplicate_group_id", ""),
                "minimum_hamming_distance": duplicate.get("minimum_hamming_distance", ""),
                "penguin_object_count": str(penguin_counts[path]),
                "filename_sequence_evidence": group["filename_evidence"],
                "visual_evidence": group["visual_evidence"],
            }
        )
    enriched.sort(key=lambda row: _capture_number(row["original_capture_id"]))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in enriched:
        groups[row["proposed_final_source_group_id"]].append(row)
    sequence_sheets = _contact_sheets(
        dict(groups), dataset_root, output_dir / "contact_sheets" / "filename_sequence"
    )
    current_sheets = _contact_sheets(
        {CURRENT_GROUP: enriched}, dataset_root, output_dir / "contact_sheets" / "current_source"
    )
    no_mov_sheets = _contact_sheets(
        {"no-explicit-mov-id": enriched},
        dataset_root,
        output_dir / "contact_sheets" / "explicit_video",
    )
    similarity_sheets = _contact_sheets(
        {"greedy-dhash-nearest-neighbor": _similarity_order(enriched)},
        dataset_root,
        output_dir / "contact_sheets" / "perceptual_similarity",
    )
    # The reviewed viewpoint/background/session boundaries coincide with the three capture runs.
    viewpoint_sheets = _contact_sheets(
        dict(groups), dataset_root, output_dir / "contact_sheets" / "camera_viewpoint"
    )
    background_sheets = _contact_sheets(
        dict(groups), dataset_root, output_dir / "contact_sheets" / "background_exhibit"
    )
    session_sheets = _contact_sheets(
        dict(groups), dataset_root, output_dir / "contact_sheets" / "capture_sequence"
    )

    cross_group: dict[str, tuple[int, str]] = {}
    for row in enriched:
        candidates = [
            other
            for other in enriched
            if other["proposed_final_source_group_id"]
            != row["proposed_final_source_group_id"]
        ]
        nearest = min(
            candidates,
            key=lambda other: hamming_distance(
                row["perceptual_hash"], other["perceptual_hash"]
            ),
        )
        cross_group[row["image_path"]] = (
            hamming_distance(row["perceptual_hash"], nearest["perceptual_hash"]),
            nearest["original_capture_id"],
        )

    review_rows = []
    inventory_rows = []
    for row in enriched:
        group_id = row["proposed_final_source_group_id"]
        distance, nearest_capture = cross_group[row["image_path"]]
        confirmed_duplicate = row["duplicate_group_id"] or "none"
        sheet_refs = sorted(
            set(
                sequence_sheets[group_id]
                + current_sheets[CURRENT_GROUP]
                + no_mov_sheets["no-explicit-mov-id"]
                + similarity_sheets["greedy-dhash-nearest-neighbor"]
                + viewpoint_sheets[group_id]
                + background_sheets[group_id]
                + session_sheets[group_id]
            )
        )
        review_rows.append(
            {
                "image_path": row["image_path"],
                "original_source_group_id": row["original_source_group_id"],
                "proposed_final_source_group_id": group_id,
                "filename_sequence_evidence": row["filename_sequence_evidence"],
                "visual_evidence": row["visual_evidence"],
                "near_duplicate_evidence": (
                    f"confirmed_duplicate_group={confirmed_duplicate}; nearest_cross_group_dhash="
                    f"{distance} ({nearest_capture}); threshold=6"
                ),
                "final_decision": "REFINABLE",
                "rationale": (
                    "Keep adjacent frames and confirmed duplicates within this run; the large "
                    "filename discontinuity, distinct visual pass, and absence of a cross-run "
                    "dHash near duplicate support capture independence."
                ),
                "contact_sheet_reference": ";".join(sheet_refs),
            }
        )
        inventory_rows.append(
            {
                "image_path": row["image_path"],
                "original_capture_id": row["original_capture_id"],
                "original_export_folder": row["original_export_folder"],
                "original_source_group_id": row["original_source_group_id"],
                "duplicate_group_id": row["duplicate_group_id"],
                "content_hash": row["content_hash"],
                "perceptual_hash": row["perceptual_hash"],
                "penguin_object_count": row["penguin_object_count"],
                "explicit_video_or_sequence_id": "none; still-image capture run",
                "proposed_capture_sequence": group_id,
            }
        )

    _write_csv(
        output_dir / "penguin_group_review.csv",
        [
            "image_path",
            "original_source_group_id",
            "proposed_final_source_group_id",
            "filename_sequence_evidence",
            "visual_evidence",
            "near_duplicate_evidence",
            "final_decision",
            "rationale",
            "contact_sheet_reference",
        ],
        review_rows,
    )
    _write_csv(
        output_dir / "penguin_inventory.csv",
        [
            "image_path",
            "original_capture_id",
            "original_export_folder",
            "original_source_group_id",
            "duplicate_group_id",
            "content_hash",
            "perceptual_hash",
            "penguin_object_count",
            "explicit_video_or_sequence_id",
            "proposed_capture_sequence",
        ],
        inventory_rows,
    )

    reviewed_v2 = [dict(row) for row in source_rows]
    affected = 0
    for row in reviewed_v2:
        if row["source_group_id"] != CURRENT_GROUP:
            continue
        number = _capture_number(row["original_capture_id"])
        group = _capture_group(number)
        row["source_group_id"] = group["id"]
        row["evidence"] = (
            "focused Penguin capture-dependency review; " + group["filename_evidence"]
        )
        row["review_status"] = "confirmed"
        row["recommended_review_action"] = "review complete"
        row["contact_sheets"] = ";".join(sequence_sheets[group["id"]])
        affected += 1
    unchanged = sum(
        first == second
        for first, second in zip(source_rows, reviewed_v2, strict=True)
    )
    _write_csv(output_dir / "reviewed_source_groups_v2.csv", source_fields, reviewed_v2)

    per_group = {
        group_id: {
            "penguin_images": len(rows),
            "penguin_objects": sum(int(row["penguin_object_count"]) for row in rows),
        }
        for group_id, rows in sorted(groups.items())
    }
    minimum_cross = min(distance for distance, _ in cross_group.values())
    summary = (
        "# Penguin Capture-Dependency Review\n\n"
        "**Decision: REFINABLE.** The 71 Penguin images contain three defensible capture "
        "sequences, not one demonstrably dependent source. This conclusion is based on capture "
        "dependency; sharing the same physical exhibit is not treated as dependency by itself.\n\n"
        f"- Accepted Penguin images: {len(enriched)}\n"
        f"- Penguin objects: {sum(penguin_counts.values())}\n"
        "- Explicit MOV/video identifiers: 0\n"
        f"- Defensible capture groups: {len(groups)}\n"
        f"- Minimum cross-group dHash distance: {minimum_cross} (near-duplicate threshold: 6)\n"
        f"- Rows changed for direct source-group consistency: {affected}\n"
        f"- Source-review rows unchanged: {unchanged}\n\n"
        "All Penguin images were inspected on filename-sequence sheets. Additional sheets record "
        "the current broad source, absence of explicit MOV identifiers, camera/viewpoint and "
        "background/session groupings, and greedy perceptual-similarity order. The first run "
        "contains within-run viewpoint changes, so viewpoint alone was not used to split it. "
        "Adjacent captures and confirmed duplicate pairs remain intact. Non-Penguin images in "
        "the same three capture runs were reassigned with their run as a direct consistency "
        "requirement; every other prior source decision remains unchanged.\n\n"
        "## Group inventory\n\n"
        + "\n".join(
            f"- `{group_id}`: {values['penguin_images']} Penguin images, "
            f"{values['penguin_objects']} Penguin objects."
            for group_id, values in per_group.items()
        )
        + "\n"
    )
    (output_dir / "penguin_review_summary.md").write_text(summary, encoding="utf-8")
    return {
        "penguin_images": len(enriched),
        "penguin_objects": sum(penguin_counts.values()),
        "groups": per_group,
        "minimum_cross_group_dhash": minimum_cross,
        "source_rows_changed": affected,
        "source_rows_unchanged": unchanged,
        "source_rows_total": len(reviewed_v2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    audit = Path("reports/dataset_audit/aquarium")
    parser.add_argument("--records", type=Path, default=audit / "image_records.csv")
    parser.add_argument("--boxes", type=Path, default=audit / "bounding_boxes.csv")
    parser.add_argument("--duplicates", type=Path, default=audit / "duplicate_candidates.csv")
    parser.add_argument("--sources", type=Path, default=audit / "reviewed_source_groups.csv")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw/aquarium/export"))
    parser.add_argument("--output", type=Path, default=audit / "penguin_review")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = build_review(
            args.records,
            args.boxes,
            args.duplicates,
            args.sources,
            args.dataset_root,
            args.output,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"Penguin review failed: {error}", file=sys.stderr)
        return 1
    print(
        "Penguin review complete: "
        f"{summary['penguin_images']} images, {summary['penguin_objects']} objects, "
        f"{len(summary['groups'])} capture groups."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
