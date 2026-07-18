"""Deterministic group-aware multi-label splitting and manifest freezing."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from synthdet.data.review import require_valid_review


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


def _union_by_value(
    groups: _DisjointSet, mapping: dict[str, str], included_paths: set[str]
) -> None:
    by_value: dict[str, list[str]] = defaultdict(list)
    for path, value in mapping.items():
        if path in included_paths and value:
            by_value[value].append(path)
    for members in by_value.values():
        for member in members[1:]:
            groups.union(members[0], member)


def _write_manifest(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(dataset_root: Path | None, relative_path: str) -> str:
    if dataset_root is None:
        return Path(relative_path).as_posix()
    root = dataset_root
    if root.is_absolute():
        try:
            root = root.resolve().relative_to(Path.cwd().resolve())
        except ValueError as error:
            raise ValueError(
                "Dataset root must be inside the repository to create repository-relative manifests"
            ) from error
    return (root / relative_path).as_posix()


def create_real_splits(
    records_path: Path,
    duplicates_path: Path,
    output_dir: Path,
    source_groups_path: Path | None = None,
    seed: int = 42,
    allow_unknown_source_groups: bool = False,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Create immutable 70/20/10 manifests from validated, grouped image records."""

    required_names = (
        "real_train.csv",
        "real_val.csv",
        "real_test.csv",
        "excluded.csv",
        "duplicate_groups.csv",
        "split_metadata.json",
    )
    existing = [output_dir / name for name in required_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen split outputs: "
            + ", ".join(str(path) for path in existing)
        )
    records = _read_csv(records_path)
    duplicate_rows = _read_csv(duplicates_path)
    if source_groups_path is not None and dataset_root is not None:
        require_valid_review(records_path, duplicates_path, source_groups_path, dataset_root)
    included = [row for row in records if row["inclusion_status"] == "included"]
    excluded = [row for row in records if row["inclusion_status"] != "included"]
    if not included:
        raise ValueError("No validated included images are available for splitting")
    paths = sorted(row["image_path"] for row in included)
    included_paths = set(paths)
    if {row["image_path"] for row in duplicate_rows} != included_paths:
        raise ValueError("Duplicate analysis rows must match the included image records exactly")
    pending_groups = sorted(
        {
            row["duplicate_group_id"]
            for row in duplicate_rows
            if row.get("duplicate_group_id") and row.get("review_status") == "pending"
        }
    )
    if pending_groups:
        raise ValueError(
            "Duplicate candidates require human review before splitting: "
            + ", ".join(pending_groups)
        )

    duplicate_map = {row["image_path"]: row["duplicate_group_id"] for row in duplicate_rows}
    source_map: dict[str, str] = {}
    if source_groups_path is not None:
        source_rows = _read_csv(source_groups_path)
        if "review_status" not in (source_rows[0] if source_rows else {}):
            raise ValueError("Source-group CSV must include review_status")
        pending_source_groups = sorted(
            {
                row["source_group_id"]
                for row in source_rows
                if row.get("review_status") not in {"confirmed", "not_applicable"}
            }
        )
        if pending_source_groups:
            raise ValueError(
                "Source groups require human review before splitting: "
                + ", ".join(pending_source_groups)
            )
        source_map = {row["image_path"]: row["source_group_id"].strip() for row in source_rows}
        missing = sorted(included_paths - source_map.keys())
        empty = sorted(path for path in included_paths if not source_map.get(path))
        if missing or empty:
            raise ValueError("Every included image must have a non-empty reviewed source group")
    elif not allow_unknown_source_groups:
        raise ValueError(
            "Reliable source groups were not supplied. Provide --source-groups or explicitly use "
            "--allow-unknown-source-groups after documenting the limitation."
        )
    else:
        source_map = {path: f"unknown-singleton:{path}" for path in paths}

    groups = _DisjointSet(paths)
    _union_by_value(groups, duplicate_map, included_paths)
    _union_by_value(groups, source_map, included_paths)
    components: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        components[groups.find(path)].append(path)
    by_path = {row["image_path"]: row for row in included}
    class_totals: Counter[str] = Counter()
    group_data: list[dict[str, Any]] = []
    for members in components.values():
        class_counts: Counter[str] = Counter()
        source_ids: set[str] = set()
        for path in members:
            classes = set(filter(None, by_path[path]["classes_present"].split(";")))
            class_counts.update(classes)
            class_totals.update(classes)
            source_ids.add(source_map[path])
        group_data.append(
            {
                "members": sorted(members),
                "class_counts": class_counts,
                "source_group_id": "+".join(sorted(source_ids)),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(group_data)
    group_data.sort(
        key=lambda group: (
            -sum(
                1 / class_totals[name]
                for name in group["class_counts"]
                if class_totals[name]
            ),
            -len(group["members"]),
        )
    )
    ratios = {"train": 0.70, "val": 0.20, "test": 0.10}
    assignments: dict[str, list[dict[str, Any]]] = {split: [] for split in ratios}
    split_sizes: Counter[str] = Counter()
    split_classes: dict[str, Counter[str]] = {split: Counter() for split in ratios}
    total_images = len(included)

    def score(candidate_split: str, candidate_group: dict[str, Any]) -> tuple[float, float]:
        size_error = 0.0
        class_error = 0.0
        for split, ratio in ratios.items():
            added_size = len(candidate_group["members"]) if split == candidate_split else 0
            projected_size = split_sizes[split] + added_size
            size_target = total_images * ratio
            size_error += ((projected_size - size_target) / max(size_target, 1)) ** 2
            for name, total in class_totals.items():
                added_class = (
                    candidate_group["class_counts"][name] if split == candidate_split else 0
                )
                projected_class = split_classes[split][name] + added_class
                class_target = total * ratio
                class_error += (
                    (projected_class - class_target) / max(class_target, 1)
                ) ** 2
        normalized_class_error = class_error / max(len(class_totals), 1)
        fill_ratio = split_sizes[candidate_split] / max(
            total_images * ratios[candidate_split], 1
        )
        return size_error + normalized_class_error, fill_ratio

    for group in group_data:
        selected = min(ratios, key=lambda split: score(split, group))
        assignments[selected].append(group)
        split_sizes[selected] += len(group["members"])
        split_classes[selected].update(group["class_counts"])

    output_dir.mkdir(parents=True, exist_ok=False)
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
    manifest_hashes: dict[str, str] = {}
    for split in ratios:
        rows: list[dict[str, Any]] = []
        for group in assignments[split]:
            for image_path in group["members"]:
                record = by_path[image_path]
                rows.append(
                    {
                        "image_path": _manifest_path(dataset_root, record["image_path"]),
                        "label_path": _manifest_path(dataset_root, record["label_path"]),
                        "content_hash": record["content_hash"],
                        "perceptual_hash": record["perceptual_hash"],
                        "source_group_id": group["source_group_id"],
                        "image_width": record["width"],
                        "image_height": record["height"],
                        "classes_present": record["classes_present"],
                        "object_count": record["object_count"],
                        "split": split,
                        "inclusion_status": "included",
                    }
                )
        rows.sort(key=lambda row: row["image_path"])
        filename = f"real_{split}.csv"
        manifest_hashes[filename] = _write_manifest(
            output_dir / filename, rows, manifest_fields
        )
    excluded_rows = [
        {
            "image_path": _manifest_path(dataset_root, row["image_path"]),
            "label_path": _manifest_path(dataset_root, row["label_path"]),
            "exclusion_reasons": row["exclusion_reasons"],
            "inclusion_status": "excluded",
        }
        for row in excluded
    ]
    manifest_hashes["excluded.csv"] = _write_manifest(
        output_dir / "excluded.csv",
        sorted(excluded_rows, key=lambda row: row["image_path"]),
        ["image_path", "label_path", "exclusion_reasons", "inclusion_status"],
    )
    duplicate_fields = list(duplicate_rows[0]) if duplicate_rows else []
    frozen_duplicate_rows = [dict(row) for row in duplicate_rows]
    for row in frozen_duplicate_rows:
        row["image_path"] = _manifest_path(dataset_root, row["image_path"])
    manifest_hashes["duplicate_groups.csv"] = _write_manifest(
        output_dir / "duplicate_groups.csv", frozen_duplicate_rows, duplicate_fields
    )
    combined_material = "\n".join(
        f"{name}:{manifest_hashes[name]}" for name in sorted(manifest_hashes)
    ).encode()
    metadata: dict[str, Any] = {
        "status": "frozen",
        "seed": seed,
        "target_ratios": ratios,
        "actual_counts": dict(split_sizes),
        "actual_ratios": {
            split: split_sizes[split] / total_images for split in ratios
        },
        "ratio_deviation": {
            split: split_sizes[split] / total_images - ratios[split] for split in ratios
        },
        "group_count": len(group_data),
        "source_group_count_per_split": {
            split: len(assignments[split]) for split in ratios
        },
        "object_count_per_split": {
            split: sum(
                int(by_path[path]["object_count"])
                for group in assignments[split]
                for path in group["members"]
            )
            for split in ratios
        },
        "image_count_per_class_per_split": {
            split: dict(sorted(split_classes[split].items())) for split in ratios
        },
        "class_coverage_limitations": {
            split: sorted(name for name in class_totals if split_classes[split][name] == 0)
            for split in ratios
        },
        "source_group_method": (
            "reviewed_mapping"
            if source_groups_path is not None
            else "documented_singleton_fallback"
        ),
        "stratification": (
            "deterministic greedy group allocation using image-level multi-label classes"
        ),
        "manifest_sha256": manifest_hashes,
        "combined_split_sha256": hashlib.sha256(combined_material).hexdigest(),
        "input_sha256": {
            "image_records.csv": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "duplicate_candidates.csv": hashlib.sha256(duplicates_path.read_bytes()).hexdigest(),
            "reviewed_source_groups.csv": (
                hashlib.sha256(source_groups_path.read_bytes()).hexdigest()
                if source_groups_path is not None
                else None
            ),
        },
        "test_restrictions": [
            "model_evaluation_only",
            "not_for_model_selection",
            "not_for_synthetic_sources",
            "not_for_synthetic_backgrounds",
        ],
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def preview_real_splits(**kwargs: Any) -> dict[str, Any]:
    """Generate a disposable candidate split without creating frozen manifests."""

    with tempfile.TemporaryDirectory(prefix="synthdet-split-preview-") as temporary:
        return create_real_splits(output_dir=Path(temporary) / "candidate", **kwargs)


def verify_real_split_reproduction(frozen_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """Reproduce a frozen split in temporary storage and compare its immutable identity."""

    metadata_path = frozen_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Frozen split metadata not found: {metadata_path}")
    frozen = json.loads(metadata_path.read_text(encoding="utf-8"))
    reproduced = preview_real_splits(**kwargs)
    if reproduced["combined_split_sha256"] != frozen.get("combined_split_sha256"):
        raise ValueError(
            "Seed/configuration reproduction mismatch: "
            f"expected {frozen.get('combined_split_sha256')}, "
            f"got {reproduced['combined_split_sha256']}"
        )
    if reproduced["manifest_sha256"] != frozen.get("manifest_sha256"):
        raise ValueError("Reproduced manifest hashes do not match the frozen metadata")
    return reproduced
