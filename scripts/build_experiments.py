"""Build, freeze, reproduce, and materialize the five controlled Aquarium regimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.config.loader import load_config  # noqa: E402
from synthdet.synthetic.contracts import (  # noqa: E402
    load_synthetic_config,
    read_csv,
    sha256_file,
    verify_active_split,
)
from synthdet.synthetic.validation import (  # noqa: E402
    validate_object_bank,
    validate_synthetic_pool,
)
from synthdet.training.experiments import (  # noqa: E402
    construct_regimes,
    freeze_experiment_design,
    load_regimes,
    validate_regimes,
    verify_experiment_reproduction,
)
from synthdet.training.materialize import materialize_views  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-frozen", action="store_true")
    action.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    return parser


def _verified_inputs(project_config_path: Path):
    project = load_config(project_config_path)
    synthetic = load_synthetic_config(PROJECT_ROOT / project.synthetic.config)
    split_dir = PROJECT_ROOT / project.dataset.paths.train_manifest.parent
    verify_active_split(split_dir, project.synthetic.active_real_split_identity)
    generation = json.loads(
        (PROJECT_ROOT / project.synthetic.manifests / "generation_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if generation["combined_synthetic_pool_identity"] != project.synthetic.pool_identity:
        raise ValueError("Frozen synthetic pool identity differs from project configuration")
    if generation["object_bank_identity"] != project.synthetic.object_bank_identity:
        raise ValueError("Frozen object-bank identity differs from project configuration")
    if generation["configuration_hash"] != sha256_file(PROJECT_ROOT / project.synthetic.config):
        raise ValueError("Frozen generator configuration identity mismatch")
    bank_errors = validate_object_bank(
        synthetic,
        PROJECT_ROOT,
        PROJECT_ROOT / project.synthetic.output,
        PROJECT_ROOT / project.synthetic.manifests,
    )
    pool_errors = validate_synthetic_pool(
        synthetic,
        PROJECT_ROOT,
        PROJECT_ROOT / project.synthetic.output,
        project.synthetic.output,
        PROJECT_ROOT / project.synthetic.manifests,
        PROJECT_ROOT / project.synthetic.manifests,
        project.synthetic.pool_size,
    )
    if bank_errors or pool_errors:
        raise ValueError(
            "Frozen synthetic validation failed:\n- "
            + "\n- ".join(bank_errors + pool_errors)
        )
    return project, synthetic, generation, split_dir


def main() -> int:
    args = build_parser().parse_args()
    try:
        project, synthetic, generation, split_dir = _verified_inputs(
            PROJECT_ROOT / args.project_config
        )
        real_rows = read_csv(split_dir / "real_train.csv")
        validation_rows = read_csv(split_dir / "real_val.csv")
        test_rows = read_csv(split_dir / "real_test.csv")
        synthetic_rows = read_csv(
            PROJECT_ROOT / project.synthetic.manifests / "synthetic_images.csv"
        )
        manifest_dir = PROJECT_ROOT / "manifests/aquarium/experiments/v1"
        if args.verify_frozen:
            result = verify_experiment_reproduction(
                manifest_dir,
                real_rows,
                synthetic_rows,
                project.dataset.class_names,
                PROJECT_ROOT,
                project.seed,
            )
            print(
                "Experiment design reproduced: "
                + result["combined_experiment_design_identity"]
            )
            return 0
        if args.materialize_only:
            regimes = load_regimes(manifest_dir)
        else:
            regimes = construct_regimes(
                real_rows,
                synthetic_rows,
                project.dataset.class_names,
                PROJECT_ROOT,
                project.seed,
                project.synthetic.active_real_split_identity,
                project.synthetic.pool_identity,
            )
            errors = validate_regimes(
                regimes,
                real_rows,
                validation_rows,
                test_rows,
                synthetic_rows,
                PROJECT_ROOT,
                project.synthetic.active_real_split_identity,
                project.synthetic.pool_identity,
            )
            if errors:
                raise ValueError("Experiment design validation failed:\n- " + "\n- ".join(errors))
            metadata = freeze_experiment_design(
                manifest_dir,
                regimes,
                project.synthetic.active_real_split_identity,
                project.synthetic.pool_identity,
                project.synthetic.object_bank_identity,
                generation["configuration_hash"],
                project.seed,
                len(validation_rows),
                PROJECT_ROOT,
                errors,
            )
            print("Frozen experiment identity: " + metadata["combined_experiment_design_identity"])
        summary = materialize_views(
            regimes,
            validation_rows,
            PROJECT_ROOT / "datasets/experiments/aquarium/v1",
            PROJECT_ROOT,
            project.dataset.class_names,
            project.seed,
        )
        print("Materialized regimes: " + ", ".join(summary))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
