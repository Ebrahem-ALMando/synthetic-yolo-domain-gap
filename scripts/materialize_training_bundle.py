"""Materialize frozen YOLO views from a training-only bundle without generation caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.config.loader import load_config  # noqa: E402
from synthdet.synthetic.contracts import read_csv, verify_active_split  # noqa: E402
from synthdet.training.bundle import validate_extracted_bundle  # noqa: E402
from synthdet.training.colab import resolve_expected_revision  # noqa: E402
from synthdet.training.experiments import load_regimes  # noqa: E402
from synthdet.training.materialize import materialize_views  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> int:
    build_parser().parse_args()
    try:
        validate_extracted_bundle(PROJECT_ROOT)
        revision = resolve_expected_revision(PROJECT_ROOT)
        print(f"Materializing validated bundle revision: {revision}")
        project = load_config(PROJECT_ROOT / "configs/project.yaml")
        split_dir = PROJECT_ROOT / project.dataset.paths.train_manifest.parent
        verify_active_split(split_dir, project.synthetic.active_real_split_identity)
        manifest_dir = PROJECT_ROOT / project.experiments.manifests
        metadata = json.loads(
            (manifest_dir / "experiment_metadata.json").read_text(encoding="utf-8")
        )
        if metadata["combined_experiment_design_identity"] != project.experiments.design_identity:
            raise ValueError("Frozen experiment identity differs from project configuration")
        summary = materialize_views(
            load_regimes(manifest_dir),
            read_csv(PROJECT_ROOT / project.dataset.paths.validation_manifest),
            PROJECT_ROOT / project.experiments.dataset_views,
            PROJECT_ROOT,
            project.dataset.class_names,
            project.seed,
        )
        print("Materialized bundle regimes: " + ", ".join(summary))
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
