"""Conservative source-group proposals and visual review contact sheets."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def original_capture_id(image_path: str) -> str:
    stem = Path(image_path).stem
    original = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    return re.sub(r"_(?:jpeg_)?jpg$|_png$", "", original)


def _contact_sheets(
    groups: dict[str, list[str]],
    dataset_root: Path,
    output_dir: Path,
    max_images_per_page: int = 20,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: dict[str, list[str]] = defaultdict(list)
    tile_width, image_height, label_height, columns = 260, 180, 44, 4
    tile_height = image_height + label_height
    for group_id, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for page_index in range(0, len(members), max_images_per_page):
            page = members[page_index : page_index + max_images_per_page]
            rows = math.ceil(len(page) / columns)
            canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
            draw = ImageDraw.Draw(canvas)
            for index, relative_path in enumerate(page):
                with Image.open(dataset_root / relative_path) as source:
                    preview = ImageOps.contain(
                        source.convert("RGB"), (tile_width - 12, image_height - 12)
                    )
                column, row = index % columns, index // columns
                x = column * tile_width + (tile_width - preview.width) // 2
                y = row * tile_height + (image_height - preview.height) // 2
                canvas.paste(preview, (x, y))
                label = original_capture_id(relative_path)
                draw.text(
                    (column * tile_width + 6, row * tile_height + image_height + 2),
                    label,
                    fill="black",
                )
                split_name = relative_path.split("/", maxsplit=1)[0]
                draw.text(
                    (column * tile_width + 6, row * tile_height + image_height + 20),
                    f"export split: {split_name}",
                    fill="#555555",
                )
            page_number = page_index // max_images_per_page + 1
            target = output_dir / f"{group_id}_page-{page_number:02d}.jpg"
            canvas.save(target, quality=90)
            sheet_paths[group_id].append(target.as_posix())
    return dict(sheet_paths)


def generate_duplicate_contact_sheets(
    duplicates_path: Path, dataset_root: Path, output_dir: Path
) -> dict[str, list[str]]:
    rows = _read_csv(duplicates_path)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("duplicate_group_id"):
            groups[row["duplicate_group_id"]].append(row["image_path"])
    return _contact_sheets(dict(groups), dataset_root, output_dir)


def propose_source_groups(
    records_path: Path,
    duplicates_path: Path,
    dataset_root: Path,
    output_path: Path,
    contact_sheet_dir: Path,
) -> dict[str, Any]:
    """Propose conservative groups without claiming unresolved provenance."""

    records = [row for row in _read_csv(records_path) if row["inclusion_status"] == "included"]
    duplicate_rows = _read_csv(duplicates_path)
    paths = sorted(row["image_path"] for row in records)
    if not paths:
        raise ValueError("No included records are available for source grouping")
    groups = _DisjointSet(paths)
    original_ids = {path: original_capture_id(path) for path in paths}

    video_members: dict[str, list[str]] = defaultdict(list)
    still_by_number: dict[int, str] = {}
    for path, original_id in original_ids.items():
        video_match = re.fullmatch(r"(IMG_\d+_MOV)-\d+", original_id)
        number_match = re.fullmatch(r"IMG_(\d+)", original_id)
        if video_match:
            video_members[video_match.group(1)].append(path)
        elif number_match:
            still_by_number[int(number_match.group(1))] = path
    for members in video_members.values():
        for member in members[1:]:
            groups.union(members[0], member)
    numbers = sorted(still_by_number)
    for first, second in zip(numbers, numbers[1:], strict=False):
        if second - first == 1:
            groups.union(still_by_number[first], still_by_number[second])

    duplicate_by_group: dict[str, list[str]] = defaultdict(list)
    for row in duplicate_rows:
        if row.get("duplicate_group_id"):
            duplicate_by_group[row["duplicate_group_id"]].append(row["image_path"])
    for members in duplicate_by_group.values():
        for member in members[1:]:
            groups.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        components[groups.find(path)].append(path)
    ordered_components = sorted(
        (sorted(members) for members in components.values()), key=lambda members: members[0]
    )
    source_id_for_path: dict[str, str] = {}
    group_status: dict[str, str] = {}
    group_evidence: dict[str, str] = {}
    group_action: dict[str, str] = {}
    contact_groups: dict[str, list[str]] = {}
    for index, members in enumerate(ordered_components, start=1):
        group_id = f"source-{index:04d}"
        member_video_bases = {
            match.group(1)
            for path in members
            if (match := re.fullmatch(r"(IMG_\d+_MOV)-\d+", original_ids[path]))
        }
        is_explicit_video = len(member_video_bases) == 1 and all(
            "_MOV-" in original_ids[path] for path in members
        )
        duplicate_ids = sorted(
            group_id_value
            for group_id_value, duplicate_members in duplicate_by_group.items()
            if set(members) & set(duplicate_members)
        )
        if is_explicit_video:
            status = "confirmed"
            evidence = "shared explicit MOV filename base"
            action = "keep all frames together"
        elif len(members) > 1:
            status = "pending"
            evidence_parts = ["consecutive IMG capture numbers"]
            if duplicate_ids:
                evidence_parts.append("dHash duplicate candidate: " + ";".join(duplicate_ids))
            evidence = "; ".join(evidence_parts)
            action = "review contact sheets; confirm or split when scene/capture changes"
        else:
            status = "pending"
            evidence = "no reliable per-image provenance; singleton proposal only"
            action = "confirm independent source or merge with a related reviewed group"
        for path in members:
            source_id_for_path[path] = group_id
        group_status[group_id] = status
        group_evidence[group_id] = evidence
        group_action[group_id] = action
        if status == "pending" and len(members) > 1:
            contact_groups[group_id] = members

    sheet_paths = _contact_sheets(contact_groups, dataset_root, contact_sheet_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "original_capture_id",
        "source_group_id",
        "evidence",
        "review_status",
        "recommended_review_action",
        "contact_sheets",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for path in paths:
            group_id = source_id_for_path[path]
            writer.writerow(
                {
                    "image_path": path,
                    "original_capture_id": original_ids[path],
                    "source_group_id": group_id,
                    "evidence": group_evidence[group_id],
                    "review_status": group_status[group_id],
                    "recommended_review_action": group_action[group_id],
                    "contact_sheets": ";".join(sheet_paths.get(group_id, [])),
                }
            )
    pending_groups = sorted(
        group_id for group_id, status in group_status.items() if status == "pending"
    )
    summary: dict[str, Any] = {
        "included_images": len(paths),
        "proposed_source_groups": len(ordered_components),
        "confirmed_video_groups": sum(status == "confirmed" for status in group_status.values()),
        "pending_groups": len(pending_groups),
        "pending_group_ids": pending_groups,
        "contact_sheet_files": sum(len(paths) for paths in sheet_paths.values()),
        "limitations": (
            "Roboflow export splits and IMG number adjacency do not establish aquarium, scene, or "
            "capture provenance. All non-MOV proposals remain pending human review."
        ),
    }
    summary_path = output_path.with_name("source_grouping_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    instructions = """# Aquarium Human Review Instructions

## Duplicate candidates

Review `duplicate_candidates.csv` alongside `duplicate_contact_sheets/`.

- `pending`: unresolved; splitting is blocked.
- `confirmed`: the images are the same capture or a near-identical sequence and must remain in one
  duplicate group.
- `rejected`: the perceptual-hash match is a false positive. Clear `duplicate_group_id` for every
  row in that rejected group so it is no longer grouped.

## Source groups

Review `reviewed_source_groups.csv` alongside `source_contact_sheets/`.

- `confirmed`: the `source_group_id` is accepted and all rows with that ID must remain together.
- `pending`: provenance or the proposed boundary is unresolved; splitting is blocked.
- `split_required`: the proposed group spans multiple scenes or captures. Assign new stable
  `source_group_id` values at the reviewed boundaries, then mark every resulting row `confirmed`.
- `merge_required`: rows belong with another proposed group. Give all affected rows one stable
  `source_group_id`, then mark them `confirmed`.

The Roboflow `train`, `valid`, and `test` folders are export placements, not source provenance.
Do not use those folder names to approve or divide source groups. Do not mark a group confirmed when
the contact sheet still shows unresolved scene changes.
"""
    output_path.with_name("review_instructions.md").write_text(instructions, encoding="utf-8")
    return summary
