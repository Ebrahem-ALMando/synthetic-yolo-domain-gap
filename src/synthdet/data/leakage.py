"""Hard-fail checks protecting the immutable real test set."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _overlap(values: dict[str, set[str]], value_name: str) -> list[str]:
    errors: list[str] = []
    names = sorted(values)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = sorted(value for value in values[first] & values[second] if value)
            if shared:
                errors.append(
                    f"{value_name} overlap between {first} and {second}: {', '.join(shared)}"
                )
    return errors


def validate_leakage(
    manifest_dir: Path,
    synthetic_source_manifests: list[Path] | None = None,
    synthetic_background_manifests: list[Path] | None = None,
    synthetic_image_manifests: list[Path] | None = None,
    expected_split_identity: str | None = None,
) -> list[str]:
    """Return every detected split/test leakage violation."""

    split_rows = {
        split: _read_csv(manifest_dir / f"real_{split}.csv") for split in ("train", "val", "test")
    }
    errors: list[str] = []
    errors.extend(
        _overlap(
            {
                split: {row.get("image_path", "") for row in rows}
                for split, rows in split_rows.items()
            },
            "image path",
        )
    )
    errors.extend(
        _overlap(
            {
                split: {row.get("content_hash", "") for row in rows}
                for split, rows in split_rows.items()
            },
            "content hash",
        )
    )
    errors.extend(
        _overlap(
            {
                split: {row.get("source_group_id", "") for row in rows}
                for split, rows in split_rows.items()
            },
            "source group",
        )
    )

    duplicate_rows = _read_csv(manifest_dir / "duplicate_groups.csv")
    duplicate_by_image = {
        row["image_path"]: row.get("duplicate_group_id", "") for row in duplicate_rows
    }
    split_duplicate_groups: dict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            group_id = duplicate_by_image.get(row["image_path"], "")
            if group_id:
                split_duplicate_groups[split].add(group_id)
    errors.extend(_overlap(split_duplicate_groups, "duplicate group"))

    train_by_path = {
        row.get("image_path", ""): row.get("content_hash", "") for row in split_rows["train"]
    }
    protected = {
        split: {
            "paths": {row.get("image_path", "") for row in split_rows[split]},
            "hashes": {row.get("content_hash", "") for row in split_rows[split]},
        }
        for split in ("val", "test")
    }
    future_manifests = [("synthetic source", path) for path in synthetic_source_manifests or []] + [
        ("synthetic background", path) for path in synthetic_background_manifests or []
    ]
    for manifest_type, path in future_manifests:
        rows = _read_csv(path)
        manifest_paths = {row.get("image_path", "") for row in rows}
        manifest_hashes = {row.get("content_hash", "") for row in rows}
        for split, values in protected.items():
            label = "Validation" if split == "val" else "Test"
            shared_paths = sorted(values["paths"] & manifest_paths)
            shared_hashes = sorted(values["hashes"] & manifest_hashes)
            if shared_paths:
                errors.append(
                    f"{label} image path present in {manifest_type} manifest {path}: "
                    + ", ".join(shared_paths)
                )
            if shared_hashes:
                errors.append(
                    f"{label} content hash present in {manifest_type} manifest {path}: "
                    + ", ".join(shared_hashes)
                )
        for row in rows:
            image_path = row.get("image_path", "")
            content_hash = row.get("content_hash", "")
            if image_path not in train_by_path:
                errors.append(
                    f"Non-train image path in {manifest_type} manifest {path}: {image_path}"
                )
            elif train_by_path[image_path] != content_hash:
                errors.append(
                    f"Train content hash mismatch in {manifest_type} manifest {path}: {image_path}"
                )

    real_paths = {row.get("image_path", "") for rows in split_rows.values() for row in rows}
    for path in synthetic_image_manifests or []:
        for row in _read_csv(path):
            output_path = row.get("synthetic_image_path", row.get("image_path", ""))
            if output_path in real_paths:
                errors.append(f"Synthetic output path collides with a real image: {output_path}")

    metadata_path = manifest_dir / "split_metadata.json"
    if not metadata_path.is_file():
        errors.append(f"Split metadata is missing: {metadata_path}")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_hashes = metadata.get("manifest_sha256", {})
        actual_hashes: dict[str, str] = {}
        for name, expected in expected_hashes.items():
            path = manifest_dir / name
            if not path.is_file():
                errors.append(f"Frozen manifest is missing: {path}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            actual_hashes[name] = actual
            if actual != expected:
                errors.append(f"Frozen manifest hash mismatch: {name}")
        if len(actual_hashes) == len(expected_hashes):
            combined_material = "\n".join(
                f"{name}:{actual_hashes[name]}" for name in sorted(actual_hashes)
            ).encode()
            actual_identity = hashlib.sha256(combined_material).hexdigest()
            if actual_identity != metadata.get("combined_split_sha256"):
                errors.append("Frozen combined split identity mismatch")
        if (
            expected_split_identity is not None
            and metadata.get("combined_split_sha256") != expected_split_identity
        ):
            errors.append(
                "Configured active split identity mismatch: "
                f"expected {expected_split_identity}, "
                f"got {metadata.get('combined_split_sha256')}"
            )
    return errors
