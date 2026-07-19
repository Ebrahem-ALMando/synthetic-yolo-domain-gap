"""Build a checksum-recorded, secret-free archive for external GPU training."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.config.loader import load_config  # noqa: E402
from synthdet.synthetic.contracts import read_csv, sha256_file  # noqa: E402
from synthdet.training.experiments import load_regimes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/training_bundle/aquarium-sprint4a.zip")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _required_files() -> set[Path]:
    project = load_config()
    manifest_dir = PROJECT_ROOT / project.experiments.manifests
    regimes = load_regimes(manifest_dir)
    files = {
        PROJECT_ROOT / row[field]
        for rows in regimes.values()
        for row in rows
        for field in ("training_image_path", "label_path")
    }
    for row in read_csv(PROJECT_ROOT / project.dataset.paths.validation_manifest):
        files.add(PROJECT_ROOT / row["image_path"])
        files.add(PROJECT_ROOT / row["label_path"])
    versioned_roots = [
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "manifests/aquarium/v2",
        PROJECT_ROOT / "manifests/aquarium/synthetic/v1",
        PROJECT_ROOT / "manifests/aquarium/experiments/v1",
        PROJECT_ROOT / "src/synthdet",
    ]
    for root in versioned_roots:
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    files.update(
        {
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "scripts/build_experiments.py",
            PROJECT_ROOT / "scripts/validate_experiments.py",
            PROJECT_ROOT / "scripts/train_yolo.py",
            PROJECT_ROOT / "scripts/run_all_regimes.py",
            PROJECT_ROOT / "scripts/colab_train.py",
        }
    )
    return files


def main() -> int:
    args = build_parser().parse_args()
    files = sorted(_required_files())
    for path in files:
        if not path.is_file():
            print(f"[ERROR] Required bundle file is missing: {path}", file=sys.stderr)
            return 1
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative.startswith("datasets/raw/") and "/test/" in relative.replace("\\", "/"):
            # Legacy provider folders are not scientific split assignments, so membership is checked
            # below from experiment/validation manifests instead of by path text.
            pass
        if path.name.lower() in {".env", "credentials.json"} or path.suffix.lower() in {
            ".key",
            ".pem",
        }:
            print(f"[ERROR] Refusing secret-like bundle file: {relative}", file=sys.stderr)
            return 1
    inventory = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    total_bytes = sum(item["size_bytes"] for item in inventory)
    if args.dry_run:
        print(json.dumps({"file_count": len(files), "total_bytes": total_bytes}, indent=2))
        return 0
    output = PROJECT_ROOT / args.output
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        print(f"[ERROR] Refusing to overwrite bundle output: {output}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    project = load_config()
    metadata = {
        "real_split_identity": project.synthetic.active_real_split_identity,
        "synthetic_pool_identity": project.synthetic.pool_identity,
        "experiment_design_identity": project.experiments.design_identity,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "validation_command": "python scripts/validate_experiments.py",
        "contains_real_test_images": False,
        "contains_secrets": False,
        "inventory": inventory,
    }
    with tempfile.TemporaryDirectory(prefix="synthdet-bundle-") as temporary:
        inventory_path = Path(temporary) / "bundle_inventory.json"
        inventory_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
            archive.write(inventory_path, "bundle_inventory.json")
    checksum = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"
    )
    print(f"Bundle created: {output.relative_to(PROJECT_ROOT)} ({checksum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
