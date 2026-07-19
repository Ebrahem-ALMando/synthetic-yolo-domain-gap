"""Checksummed, test-safe transfer bundles for external CUDA training."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from synthdet.config.loader import load_config
from synthdet.synthetic.contracts import read_csv, sha256_file, stable_json_hash
from synthdet.training.experiments import load_regimes

BUNDLE_VERSION = "aquarium-sprint4b-training-bundle-v1"
SECRET_NAMES = {".env", "credentials.json", "service_account.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Git HEAD is unavailable or is not a full commit SHA")
    return revision


def git_branch(root: Path) -> str:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not branch:
        raise ValueError("Git branch is unavailable; detached HEAD cannot build a bundle")
    return branch


def git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def clean_source_state(root: Path) -> dict[str, Any]:
    branch = git_branch(root)
    revision = git_revision(root)
    dirty = git_dirty(root)
    if branch != "main":
        raise ValueError(f"Training bundles must be built from main, not {branch!r}")
    if dirty:
        raise ValueError("Training bundles require a clean Git worktree")
    return {
        "expected_repository_revision": revision,
        "source_branch": branch,
        "source_worktree_dirty": False,
    }


def required_bundle_files(root: Path) -> list[Path]:
    project = load_config(root / "configs/project.yaml")
    regimes = load_regimes(root / project.experiments.manifests)
    files = {
        root / row[field]
        for rows in regimes.values()
        for row in rows
        for field in ("training_image_path", "label_path")
    }
    for row in read_csv(root / project.dataset.paths.validation_manifest):
        files.add(root / row["image_path"])
        files.add(root / row["label_path"])
    versioned_roots = (
        "configs",
        "manifests/aquarium/v2",
        "manifests/aquarium/synthetic/v1",
        "manifests/aquarium/experiments/v1",
        "src/synthdet",
        "scripts",
    )
    for relative_root in versioned_roots:
        files.update(
            path
            for path in (root / relative_root).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    notebook = root / "notebooks/sprint4b_full_training_colab.ipynb"
    if notebook.is_file():
        files.add(notebook)
    files.update(
        {
            root / "pyproject.toml",
            root / "README.md",
            root / "docs/training_protocol.md",
            root / "docs/reproducibility.md",
        }
    )
    result = sorted(files)
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Required bundle file is missing: {path}")
        if path.name.lower() in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
            raise ValueError(f"Secret-like file is prohibited from the bundle: {path}")
        if ".venv" in path.parts or "artifacts" in path.parts or "__pycache__" in path.parts:
            raise ValueError(f"Local runtime artifact is prohibited from the bundle: {path}")
    return result


def create_inventory(
    root: Path, files: list[Path], source_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = clean_source_state(root) if source_state is None else source_state
    project = load_config(root / "configs/project.yaml")
    weight = yaml.safe_load(
        (root / "configs/training/base_weight.yaml").read_text(encoding="utf-8")
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    identity_inputs = {
        "bundle_version": BUNDLE_VERSION,
        **source,
        "real_split_identity": project.synthetic.active_real_split_identity,
        "synthetic_pool_identity": project.synthetic.pool_identity,
        "experiment_design_identity": project.experiments.design_identity,
        "base_weight_sha256": weight["sha256"],
        "files": entries,
    }
    return {
        **identity_inputs,
        "bundle_identity": stable_json_hash(identity_inputs),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "base_weight": weight,
        "validation_command": "python scripts/validate_training_bundle.py --extracted-root .",
        "contains_real_test_images": False,
        "contains_secrets": False,
        "contains_virtual_environment": False,
        "contains_smoke_or_final_runs": False,
        "test_set_access_policy": "manifest_identity_only",
        "inventory": entries,
    }


def build_bundle(root: Path, output: Path) -> dict[str, Any]:
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    inventory_path = output.with_suffix(output.suffix + ".inventory.json")
    if output.exists() or checksum_path.exists() or inventory_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle artifact: {output}")
    source = clean_source_state(root)
    files = required_bundle_files(root)
    inventory = create_inventory(root, files, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("training_bundle_inventory.json", payload)
    archive_sha256 = sha256_file(output)
    inventory["archive_name"] = output.name
    inventory["archive_sha256"] = archive_sha256
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8")
    return inventory


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            target = (destination / Path(*relative.parts)).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Archive member escapes extraction root: {member.filename}")
        archive.extractall(destination)


def validate_extracted_bundle(root: Path) -> dict[str, Any]:
    inventory_path = root / "training_bundle_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError("Extracted bundle inventory is missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    required_metadata = {
        "bundle_version",
        "expected_repository_revision",
        "source_branch",
        "source_worktree_dirty",
        "real_split_identity",
        "synthetic_pool_identity",
        "experiment_design_identity",
        "base_weight_sha256",
        "files",
        "bundle_identity",
        "inventory",
    }
    missing_metadata = sorted(required_metadata - set(inventory))
    if missing_metadata:
        raise ValueError(
            "Bundle inventory metadata is missing: " + ", ".join(missing_metadata)
        )
    for entry in inventory["inventory"]:
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"Missing bundled file: {entry['path']}")
        elif path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            errors.append(f"Bundled file hash/size mismatch: {entry['path']}")
    identity_keys = (
        "bundle_version",
        "expected_repository_revision",
        "source_branch",
        "source_worktree_dirty",
        "real_split_identity",
        "synthetic_pool_identity",
        "experiment_design_identity",
        "base_weight_sha256",
        "files",
    )
    identity_inputs = {key: inventory.get(key) for key in identity_keys}
    if stable_json_hash(identity_inputs) != inventory.get("bundle_identity"):
        errors.append("Bundle identity mismatch")
    revision = inventory.get("expected_repository_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("Expected repository revision is missing or invalid")
    if inventory.get("source_branch") != "main":
        errors.append("Bundle source branch is not main")
    if inventory.get("source_worktree_dirty") is not False:
        errors.append("Bundle was not built from a clean source worktree")
    test_rows = read_csv(root / "manifests/aquarium/v2/real_test.csv")
    protected_paths = {row["image_path"] for row in test_rows}
    protected_hashes = {row["content_hash"] for row in test_rows}
    bundled_paths = {entry["path"] for entry in inventory["inventory"]}
    collisions = sorted(protected_paths & bundled_paths)
    if collisions:
        errors.append(f"Real-test image is bundled: {collisions[0]}")
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for entry in inventory["inventory"]:
        if (
            Path(entry["path"]).suffix.lower() in image_suffixes
            and entry["sha256"] in protected_hashes
        ):
            errors.append(f"Protected real-test content hash is bundled: {entry['path']}")
            break
    prohibited_parts = {".venv", "artifacts", "runs", "__pycache__"}
    for entry in inventory["inventory"]:
        path = PurePosixPath(entry["path"])
        if prohibited_parts & set(path.parts):
            errors.append(f"Prohibited runtime path is bundled: {entry['path']}")
        if path.name.lower() in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
            errors.append(f"Secret-like file is bundled: {entry['path']}")
    if errors:
        raise ValueError("Bundle validation failed:\n- " + "\n- ".join(errors))
    return inventory
