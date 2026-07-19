"""Configuration and frozen-input contracts for synthetic generation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from synthdet.data.leakage import validate_leakage


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def repository_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {path}") from error


@dataclass(frozen=True)
class SyntheticConfig:
    path: Path
    raw: dict[str, Any]
    dataset: dict[str, Any]
    object_bank: dict[str, Any]
    sampling: dict[str, Any]
    transforms: dict[str, Any]
    placement: dict[str, Any]
    configuration_hash: str

    @property
    def class_names(self) -> list[str]:
        return list(self.dataset["class_names"])

    @property
    def split_identity(self) -> str:
        return str(self.dataset["active_split_identity"])

    @property
    def root_seed(self) -> int:
        return int(self.dataset["root_seed"])


def load_synthetic_config(path: Path) -> SyntheticConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Synthetic configuration root must be a mapping")
    required_sections = (
        "synthetic_dataset",
        "object_bank",
        "sampling",
        "transforms",
        "placement",
    )
    missing = [name for name in required_sections if not isinstance(raw.get(name), dict)]
    if missing:
        raise ValueError("Synthetic configuration is missing sections: " + ", ".join(missing))
    dataset = raw["synthetic_dataset"]
    if dataset.get("mode") != "distribution_matched_copy_paste":
        raise ValueError("Primary generator requires distribution_matched_copy_paste mode")
    if int(dataset.get("root_seed", -1)) != 42:
        raise ValueError("Primary generator root seed must be 42")
    if int(dataset.get("full_pool_size", -1)) != 427:
        raise ValueError("Primary synthetic pool must contain 427 images")
    return SyntheticConfig(
        path=path,
        raw=raw,
        dataset=dataset,
        object_bank=raw["object_bank"],
        sampling=raw["sampling"],
        transforms=raw["transforms"],
        placement=raw["placement"],
        configuration_hash=sha256_file(path),
    )


def derive_seed(root_seed: int, *parts: object) -> int:
    material = "|".join([str(root_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def verify_active_split(
    manifest_dir: Path,
    expected_identity: str,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Hard-fail unless the configured immutable real split is exactly intact."""

    if manifest_dir.as_posix().rstrip("/").split("/")[-1] != "v2":
        raise ValueError(f"Synthetic generation requires active Split V2: {manifest_dir}")
    metadata_path = manifest_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Split metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("combined_split_sha256") != expected_identity:
        raise ValueError(
            "Active split identity mismatch: "
            f"expected {expected_identity}, got {metadata.get('combined_split_sha256')}"
        )
    expected_counts = expected_counts or {"train": 427, "val": 140, "test": 68}
    for split, expected in expected_counts.items():
        rows = read_csv(manifest_dir / f"real_{split}.csv")
        if len(rows) != expected:
            raise ValueError(f"Active {split} manifest has {len(rows)} rows; expected {expected}")
    leakage_errors = validate_leakage(manifest_dir, expected_split_identity=expected_identity)
    if leakage_errors:
        raise ValueError(
            "Active split leakage/integrity failure:\n- " + "\n- ".join(leakage_errors)
        )
    return metadata
