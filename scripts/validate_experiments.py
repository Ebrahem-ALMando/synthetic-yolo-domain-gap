"""Hard-fail validation for frozen regime manifests and YOLO dataset views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.config.loader import load_config  # noqa: E402
from synthdet.synthetic.contracts import read_csv, sha256_file, verify_active_split  # noqa: E402
from synthdet.training.experiments import load_regimes, validate_regimes  # noqa: E402
from synthdet.training.materialize import validate_materialized_views  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("manifests/aquarium/experiments/v1"))
    parser.add_argument("--views", type=Path, default=Path("datasets/experiments/aquarium/v1"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        project = load_config()
        split_dir = PROJECT_ROOT / project.dataset.paths.train_manifest.parent
        verify_active_split(split_dir, project.synthetic.active_real_split_identity)
        manifest_dir = PROJECT_ROOT / args.manifests
        metadata = json.loads(
            (manifest_dir / "experiment_metadata.json").read_text(encoding="utf-8")
        )
        regimes = load_regimes(manifest_dir)
        errors = validate_regimes(
            regimes,
            read_csv(split_dir / "real_train.csv"),
            read_csv(split_dir / "real_val.csv"),
            read_csv(split_dir / "real_test.csv"),
            read_csv(PROJECT_ROOT / project.synthetic.manifests / "synthetic_images.csv"),
            PROJECT_ROOT,
            project.synthetic.active_real_split_identity,
            project.synthetic.pool_identity,
        )
        for name, expected in metadata["regime_manifest_hashes"].items():
            if sha256_file(manifest_dir / name) != expected:
                errors.append(f"Frozen regime manifest hash mismatch: {name}")
        validation_hashes = {
            row["content_hash"] for row in read_csv(split_dir / "real_val.csv")
        }
        errors.extend(
            validate_materialized_views(
                PROJECT_ROOT / args.views, project.dataset.class_names, validation_hashes
            )
        )
        if errors:
            print(
                "[ERROR] Experiment validation failed:\n- " + "\n- ".join(errors),
                file=sys.stderr,
            )
            return 1
        print("Experiment manifests, complementary pairing, and YOLO views passed validation.")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
