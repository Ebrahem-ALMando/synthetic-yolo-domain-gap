"""Generate the data-derived Sprint 4A experiment-design audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.config.loader import load_config  # noqa: E402
from synthdet.synthetic.contracts import read_csv  # noqa: E402
from synthdet.training.audit import audit_experiment_design  # noqa: E402
from synthdet.training.experiments import load_regimes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/experiment_audit/aquarium/v1"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = load_config()
    manifest_dir = PROJECT_ROOT / "manifests/aquarium/experiments/v1"
    metadata = json.loads((manifest_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    summary = audit_experiment_design(
        load_regimes(manifest_dir),
        metadata,
        read_csv(PROJECT_ROOT / project.dataset.paths.validation_manifest),
        project.dataset.class_names,
        PROJECT_ROOT,
        PROJECT_ROOT / args.output,
    )
    print("Experiment audit created: " + summary["combined_experiment_design_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
