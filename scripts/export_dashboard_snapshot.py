"""Export dashboard-safe repository metadata without reading protected images or model weights."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "apps/web/src/data/generated/project-snapshot.json"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def csv_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def category_counts(path: Path, field: str) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        values = Counter(row[field] for row in csv.DictReader(handle) if row.get(field))
    return dict(sorted(values.items()))


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_snapshot() -> dict[str, Any]:
    project = load_yaml(PROJECT_ROOT / "configs/project.yaml")
    common = load_yaml(PROJECT_ROOT / "configs/training/common.yaml")
    split = load_json(PROJECT_ROOT / project["dataset"]["paths"]["split_metadata"])
    generation = load_json(
        PROJECT_ROOT / project["synthetic"]["manifests"] / "generation_metadata.json"
    )
    experiments = load_json(
        PROJECT_ROOT / project["experiments"]["manifests"] / "experiment_metadata.json"
    )
    environment = load_json(PROJECT_ROOT / "reports/training_environment/sprint4a.json")
    smoke = load_json(PROJECT_ROOT / "reports/training_environment/sprint4a_smoke_summary.json")
    class_names = project["dataset"]["class_names"]
    arabic_names = {
        "fish": "سمك",
        "jellyfish": "قنديل البحر",
        "penguin": "بطريق",
        "puffin": "بفن",
        "shark": "قرش",
        "starfish": "نجم البحر",
        "stingray": "شفنين",
    }
    train_class = split["image_count_per_class_per_split"]["train"]
    class_statistics = [
        {
            "id": index,
            "key": name,
            "nameAr": arabic_names[name],
            "realTrainImages": train_class[name],
            "syntheticObjects": generation["actual_pasted_class_counts"][name],
        }
        for index, name in enumerate(class_names)
    ]
    display_names = {
        "synthetic_only": "اصطناعي فقط",
        "real_25": "25% حقيقي",
        "real_50": "50% حقيقي",
        "real_75": "75% حقيقي",
        "real_only": "حقيقي فقط",
    }
    regime_order = ["synthetic_only", "real_25", "real_50", "real_75", "real_only"]
    regimes = []
    for key in regime_order:
        ratio = experiments["realized_ratios"][key]
        config = load_yaml(PROJECT_ROOT / "configs/training/regimes" / f"{key}.yaml")
        regimes.append(
            {
                "id": key,
                "nameAr": display_names[key],
                "realCount": ratio["real_count"],
                "syntheticCount": ratio["synthetic_count"],
                "total": ratio["total"],
                "validationCount": experiments["validation_set_count"],
                "realFraction": ratio["real_fraction"],
                "manifestHash": config["expected_manifest_hash"],
                "status": "awaiting_results",
                "checkpointAvailable": False,
                "validationMetricsAvailable": False,
                "finalTestMetricsAvailable": False,
            }
        )
    object_bank_path = PROJECT_ROOT / project["synthetic"]["manifests"] / "object_bank.csv"
    failed_path = (
        PROJECT_ROOT / project["synthetic"]["manifests"] / "failed_generation_attempts.csv"
    )
    return {
        "schemaVersion": 1,
        "source": "repository",
        "gitRevision": git_revision(),
        "project": {
            "name": "SynthDet",
            "nameAr": "سينث دِت",
            "descriptionAr": "منصة تحليل البيانات الاصطناعية وكشف الأجسام",
            "sloganAr": "رؤية ذكية. بيانات اصطناعية. كشف أدق.",
            "phase": "Sprint 4B — التنفيذ الخارجي عبر CUDA",
            "dashboardStatus": "sprint6a_dashboard_foundation_implemented",
            "seed": project["seed"],
        },
        "identities": {
            "realSplit": project["synthetic"]["active_real_split_identity"],
            "syntheticPool": project["synthetic"]["pool_identity"],
            "objectBank": project["synthetic"]["object_bank_identity"],
            "generatorConfiguration": project["synthetic"][
                "generator_configuration_identity"
            ],
            "experimentDesign": project["experiments"]["design_identity"],
            "training": None,
        },
        "dataset": {
            "name": project["dataset"]["name"],
            "version": project["dataset"]["version"],
            "status": project["dataset"]["status"],
            "classCount": len(class_names),
            "classes": class_statistics,
            "splits": split["actual_counts"],
            "objects": split["object_count_per_split"],
            "sourceGroups": split["source_group_count_per_split"],
            "duplicateGroupCount": csv_count(
                PROJECT_ROOT / project["dataset"]["paths"]["duplicate_groups"]
            ),
            "leakageStatus": "passed",
            "protectedTest": True,
        },
        "synthetic": {
            "poolSize": generation["generated_count"],
            "status": generation["status"],
            "mode": generation["generation_mode"],
            "generatorVersion": generation["generator_version"],
            "pastedObjects": generation["pasted_object_count"],
            "acceptedObjectBankItems": generation["object_bank_generation_filter"][
                "eligible_generation_objects"
            ],
            "objectBankRecords": csv_count(object_bank_path),
            "objectSizes": category_counts(object_bank_path, "object_size_category"),
            "failedAttempts": csv_count(failed_path),
            "rejectedPlacements": generation["rejected_placement_attempt_count"],
        },
        "experiments": regimes,
        "training": {
            "state": "awaiting_external_cuda",
            "model": common["model"]["architecture"],
            "epochs": common["training"]["epochs"],
            "imageSize": common["training"]["imgsz"],
            "preferredBatch": common["hardware_profiles"]["standard"]["batch"],
            "fallbackBatch": common["hardware_profiles"]["low_memory"]["batch"],
            "completedFinalRegimes": 0,
            "smokeRegimesCompleted": smoke["completed_regime_count"],
            "gpu": environment["nvidia_smi"][0]["name"]
            if environment.get("nvidia_smi")
            else None,
            "profile": None,
            "testSetAccessCount": 0,
        },
        "environment": {
            "python": environment["python_version"],
            "torch": environment["torch_version"],
            "ultralytics": environment["ultralytics_version"],
            "platform": environment["operating_system"],
            "cudaAvailableLocally": environment["cuda_available"],
            "classification": environment["classification"],
        },
        "scientificResults": {
            "available": False,
            "finalTestEvaluated": False,
            "messageAr": "النتائج العلمية النهائية مقفلة حتى اكتمال التدريب والتقييم المحدد مسبقًا.",
        },
        "audit": {
            "splitFrozen": split["status"] == "frozen",
            "syntheticFrozen": generation["status"] == "frozen",
            "experimentValidationPassed": experiments["validation_results"]["passed"],
            "testSetUsedForExperiments": experiments["validation_results"]["test_set_used"],
            "protectedContentIncluded": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed_root = (PROJECT_ROOT / "apps/web/src/data/generated").resolve()
    if allowed_root not in output.parents:
        raise ValueError(f"Snapshot output must remain under {allowed_root}")
    payload = build_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "generated",
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
                "class_count": payload["dataset"]["classCount"],
                "regime_count": len(payload["experiments"]),
                "protected_content_included": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
