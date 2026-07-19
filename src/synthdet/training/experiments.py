"""Deterministic construction and validation of the five controlled regimes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthdet.synthetic.contracts import read_csv, sha256_file, stable_json_hash, write_csv

REGIME_COUNTS: dict[str, tuple[int, int]] = {
    "synthetic_only": (0, 427),
    "real_25": (107, 320),
    "real_50": (214, 213),
    "real_75": (320, 107),
    "real_only": (427, 0),
}
SELECTION_ALGORITHM = "deterministic_multilabel_deficit_greedy_v1"
MANIFEST_FIELDS = [
    "regime_name",
    "training_image_path",
    "label_path",
    "sample_type",
    "underlying_real_canvas_path",
    "source_group_id",
    "image_hash",
    "label_hash",
    "class_ids_present",
    "selection_seed",
    "real_split_identity",
    "synthetic_pool_identity",
]


def _tie(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _class_ids(row: dict[str, str], class_names: list[str]) -> set[int]:
    if "class_ids_present" in row:
        return {int(value) for value in row["class_ids_present"].split(";") if value != ""}
    name_to_id = {name: index for index, name in enumerate(class_names)}
    return {name_to_id[name] for name in row["classes_present"].split(";") if name}


def deterministic_multilabel_select(
    rows: list[dict[str, str]],
    count: int,
    class_names: list[str],
    seed: int,
    key_field: str,
) -> set[str]:
    """Select a deterministic subset while greedily minimizing class-image deficits."""

    if not 0 <= count <= len(rows):
        raise ValueError("Selection count is outside the available row count")
    if count == 0:
        return set()
    classes = {row[key_field]: _class_ids(row, class_names) for row in rows}
    totals = Counter(class_id for values in classes.values() for class_id in values)
    targets = {class_id: totals[class_id] * count / len(rows) for class_id in totals}
    selected: set[str] = set()
    selected_counts: Counter[int] = Counter()
    remaining = {row[key_field]: row for row in rows}
    while len(selected) < count:
        def score(item: tuple[str, dict[str, str]]) -> tuple[float, float, str]:
            key, _ = item
            ids = classes[key]
            deficit = sum(
                max(targets[class_id] - selected_counts[class_id], 0.0) for class_id in ids
            )
            rarity = sum(1.0 / totals[class_id] for class_id in ids)
            return (-deficit, -rarity, _tie(seed, key))

        key, _ = min(remaining.items(), key=score)
        selected.add(key)
        selected_counts.update(classes[key])
        del remaining[key]
    return selected


def _real_row(
    regime: str,
    row: dict[str, str],
    class_names: list[str],
    root: Path,
    seed: int,
    real_identity: str,
    synthetic_identity: str,
) -> dict[str, str]:
    return {
        "regime_name": regime,
        "training_image_path": row["image_path"],
        "label_path": row["label_path"],
        "sample_type": "real",
        "underlying_real_canvas_path": row["image_path"],
        "source_group_id": row["source_group_id"],
        "image_hash": row["content_hash"],
        "label_hash": sha256_file(root / row["label_path"]),
        "class_ids_present": ";".join(str(value) for value in sorted(_class_ids(row, class_names))),
        "selection_seed": str(seed),
        "real_split_identity": real_identity,
        "synthetic_pool_identity": synthetic_identity,
    }


def _synthetic_row(
    regime: str,
    row: dict[str, str],
    seed: int,
    real_identity: str,
    synthetic_identity: str,
) -> dict[str, str]:
    return {
        "regime_name": regime,
        "training_image_path": row["synthetic_image_path"],
        "label_path": row["synthetic_label_path"],
        "sample_type": "synthetic",
        "underlying_real_canvas_path": row["base_image_path"],
        "source_group_id": row["base_source_group_id"],
        "image_hash": row["output_image_hash"],
        "label_hash": row["output_label_hash"],
        "class_ids_present": row["class_ids_present"],
        "selection_seed": str(seed),
        "real_split_identity": real_identity,
        "synthetic_pool_identity": synthetic_identity,
    }


def construct_regimes(
    real_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    class_names: list[str],
    root: Path,
    seed: int,
    real_identity: str,
    synthetic_identity: str,
    regime_counts: dict[str, tuple[int, int]] = REGIME_COUNTS,
) -> dict[str, list[dict[str, str]]]:
    """Build complementary real/synthetic regimes from a one-to-one canvas mapping."""

    real_by_path = {row["image_path"]: row for row in real_rows}
    synthetic_by_base = {row["base_image_path"]: row for row in synthetic_rows}
    if len(real_by_path) != len(real_rows) or len(synthetic_by_base) != len(synthetic_rows):
        raise ValueError("Real paths and synthetic base canvases must each be unique")
    if set(real_by_path) != set(synthetic_by_base):
        raise ValueError("Synthetic pool is not a one-to-one mapping of all real-train canvases")
    regimes: dict[str, list[dict[str, str]]] = {}
    for regime, (real_count, synthetic_count) in regime_counts.items():
        if real_count + synthetic_count != len(real_rows):
            raise ValueError(f"{regime} does not match the fixed training budget")
        selected_real = deterministic_multilabel_select(
            real_rows, real_count, class_names, seed, "image_path"
        )
        rows = [
            _real_row(
                regime,
                real_by_path[path],
                class_names,
                root,
                seed,
                real_identity,
                synthetic_identity,
            )
            if path in selected_real
            else _synthetic_row(
                regime, synthetic_by_base[path], seed, real_identity, synthetic_identity
            )
            for path in sorted(real_by_path)
        ]
        if sum(row["sample_type"] == "synthetic" for row in rows) != synthetic_count:
            raise AssertionError("Complement construction produced the wrong synthetic count")
        regimes[regime] = rows
    return regimes


def validate_regimes(
    regimes: dict[str, list[dict[str, str]]],
    real_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    root: Path,
    real_identity: str,
    synthetic_identity: str,
    regime_counts: dict[str, tuple[int, int]] = REGIME_COUNTS,
) -> list[str]:
    errors: list[str] = []
    real_by_path = {row["image_path"]: row for row in real_rows}
    synthetic_by_path = {row["synthetic_image_path"]: row for row in synthetic_rows}
    protected_paths = {row["image_path"] for row in validation_rows + test_rows}
    protected_hashes = {row["content_hash"] for row in validation_rows + test_rows}
    expected_canvases = set(real_by_path)
    for regime, expected in regime_counts.items():
        rows = regimes.get(regime, [])
        real_count, synthetic_count = expected
        if len(rows) != real_count + synthetic_count:
            errors.append(f"{regime}: expected 427 rows, found {len(rows)}")
            continue
        if sum(row["sample_type"] == "real" for row in rows) != real_count:
            errors.append(f"{regime}: real count mismatch")
        if sum(row["sample_type"] == "synthetic" for row in rows) != synthetic_count:
            errors.append(f"{regime}: synthetic count mismatch")
        canvases = [row["underlying_real_canvas_path"] for row in rows]
        if len(canvases) != len(set(canvases)):
            errors.append(f"{regime}: an underlying canvas is represented more than once")
        if set(canvases) != expected_canvases:
            errors.append(f"{regime}: underlying canvas coverage is incomplete")
        for row in rows:
            image_path = row["training_image_path"]
            label_path = row["label_path"]
            if image_path in protected_paths or row["image_hash"] in protected_hashes:
                errors.append(f"{regime}: protected image used for training: {image_path}")
            if row["real_split_identity"] != real_identity:
                errors.append(f"{regime}: real identity mismatch")
            if row["synthetic_pool_identity"] != synthetic_identity:
                errors.append(f"{regime}: synthetic identity mismatch")
            source = (
                real_by_path.get(image_path)
                if row["sample_type"] == "real"
                else synthetic_by_path.get(image_path)
            )
            if source is None:
                errors.append(f"{regime}: unrecognized training path: {image_path}")
                continue
            if not (root / image_path).is_file() or not (root / label_path).is_file():
                errors.append(f"{regime}: missing image-label pair: {image_path}")
            elif sha256_file(root / image_path) != row["image_hash"]:
                errors.append(f"{regime}: image hash mismatch: {image_path}")
            elif sha256_file(root / label_path) != row["label_hash"]:
                errors.append(f"{regime}: label hash mismatch: {label_path}")
    return errors


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def freeze_experiment_design(
    output_dir: Path,
    regimes: dict[str, list[dict[str, str]]],
    real_identity: str,
    synthetic_identity: str,
    object_bank_identity: str,
    generator_configuration_identity: str,
    seed: int,
    validation_count: int,
    root: Path,
    validation_errors: list[str],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment manifests already exist: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, str] = {}
    realized: dict[str, dict[str, Any]] = {}
    for regime, rows in regimes.items():
        path = output_dir / f"{regime}.csv"
        write_csv(path, MANIFEST_FIELDS, rows)
        manifest_hashes[path.name] = sha256_file(path)
        real_count = sum(row["sample_type"] == "real" for row in rows)
        synthetic_count = len(rows) - real_count
        realized[regime] = {
            "total": len(rows),
            "real_count": real_count,
            "synthetic_count": synthetic_count,
            "real_fraction": real_count / len(rows),
            "synthetic_fraction": synthetic_count / len(rows),
        }
    identity_inputs = {
        "real_split_identity": real_identity,
        "synthetic_pool_identity": synthetic_identity,
        "object_bank_identity": object_bank_identity,
        "generator_configuration_identity": generator_configuration_identity,
        "root_seed": seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "regime_manifest_hashes": manifest_hashes,
    }
    metadata = {
        **identity_inputs,
        "combined_experiment_design_identity": stable_json_hash(identity_inputs),
        "code_revision": _git_revision(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "realized_ratios": realized,
        "validation_set_count": validation_count,
        "validation_results": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "complementary_pairing": not validation_errors,
            "test_set_used": False,
        },
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_regimes(manifest_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {name: read_csv(manifest_dir / f"{name}.csv") for name in REGIME_COUNTS}


def verify_experiment_reproduction(
    frozen_dir: Path,
    real_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    class_names: list[str],
    root: Path,
    seed: int,
) -> dict[str, Any]:
    metadata = json.loads((frozen_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    regime_counts = {
        name: (int(values["real_count"]), int(values["synthetic_count"]))
        for name, values in metadata["realized_ratios"].items()
    }
    regimes = construct_regimes(
        real_rows,
        synthetic_rows,
        class_names,
        root,
        seed,
        metadata["real_split_identity"],
        metadata["synthetic_pool_identity"],
        regime_counts,
    )
    with tempfile.TemporaryDirectory(prefix="synthdet-experiments-") as temporary:
        hashes: dict[str, str] = {}
        for name, rows in regimes.items():
            path = Path(temporary) / f"{name}.csv"
            write_csv(path, MANIFEST_FIELDS, rows)
            hashes[path.name] = sha256_file(path)
    if hashes != metadata["regime_manifest_hashes"]:
        raise ValueError("Regime manifest reproduction did not match frozen hashes")
    identity_inputs = {
        key: metadata[key]
        for key in (
            "real_split_identity",
            "synthetic_pool_identity",
            "object_bank_identity",
            "generator_configuration_identity",
            "root_seed",
            "selection_algorithm",
            "regime_manifest_hashes",
        )
    }
    identity = stable_json_hash(identity_inputs)
    if identity != metadata["combined_experiment_design_identity"]:
        raise ValueError("Experiment-design identity reproduction failed")
    return {"regime_manifest_hashes": hashes, "combined_experiment_design_identity": identity}


def read_manifest_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))
