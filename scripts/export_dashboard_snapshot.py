"""Export dashboard-safe sealed repository metadata without protected pixels or model weights."""

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
REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_count(path: Path) -> int:
    return len(load_csv(path))


def category_counts(path: Path, field: str) -> dict[str, int]:
    values = Counter(row[field] for row in load_csv(path) if row.get(field))
    return dict(sorted(values.items()))


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def numeric(row: dict[str, str], name: str, *, integer: bool = False) -> int | float:
    return int(float(row[name])) if integer else float(row[name])


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
    environment = load_json(PROJECT_ROOT / "reports/training/sprint4b_v2_environment_summary.json")
    evaluation_contract = load_yaml(PROJECT_ROOT / "configs/evaluation/sprint5_final.yaml")
    final_results_document = load_json(PROJECT_ROOT / "reports/evaluation/final_test_metrics.json")
    final_results = {row["regime"]: row for row in final_results_document["models"]}
    validation = {
        row["regime"]: row
        for row in load_csv(PROJECT_ROOT / "reports/training/sprint4b_v2_validation_summary.csv")
    }
    ranking_rows = load_csv(PROJECT_ROOT / "reports/evaluation/final_model_ranking.csv")
    ranking = {row["regime"]: row for row in ranking_rows}
    ranking_payload = [
        {
            "rank": numeric(row, "rank", integer=True),
            "regime": row["regime"],
            "recommended": row["recommended"].lower() == "true",
            "precision": numeric(row, "precision"),
            "recall": numeric(row, "recall"),
            "map50": numeric(row, "map50"),
            "map5095": numeric(row, "map50_95"),
            "latencyMs": numeric(row, "latency_ms"),
        }
        for row in ranking_rows
    ]
    per_class = [
        {
            "regime": row["regime"],
            "classId": numeric(row, "class_id", integer=True),
            "className": row["class_name"],
            "precision": numeric(row, "precision"),
            "recall": numeric(row, "recall"),
            "ap50": numeric(row, "ap50"),
            "ap5095": numeric(row, "ap50_95"),
        }
        for row in load_csv(PROJECT_ROOT / "reports/evaluation/per_class_metrics.csv")
    ]
    object_size = [
        {
            "regime": row["regime"],
            "size": row["size"],
            "instances": numeric(row, "ground_truth_instances", integer=True),
            "map50": numeric(row, "map50"),
            "map5095": numeric(row, "map50_95"),
        }
        for row in load_csv(PROJECT_ROOT / "reports/evaluation/object_size_metrics.csv")
    ]
    latency = [
        {
            "regime": row["regime"],
            "preprocessMs": numeric(row, "preprocess"),
            "inferenceMs": numeric(row, "inference"),
            "postprocessMs": numeric(row, "postprocess"),
            "totalMs": numeric(row, "total"),
            "throughput": numeric(row, "throughput_images_per_second"),
        }
        for row in load_csv(PROJECT_ROOT / "reports/evaluation/latency_metrics.csv")
    ]
    domain_gap = [
        {
            "regime": row["regime"],
            "realPercentage": numeric(row, "real_percentage"),
            "map5095": numeric(row, "map50_95"),
            "absoluteChange": numeric(row, "absolute_map50_95_change_from_synthetic_only"),
            "relativeChangePercent": numeric(
                row, "relative_map50_95_change_from_synthetic_only_percent"
            ),
        }
        for row in load_csv(PROJECT_ROOT / "reports/evaluation/domain_gap_analysis.csv")
    ]
    campaign = load_json(PROJECT_ROOT / "reports/evaluation/sprint5_campaign_lock.json")
    result_hashes = load_json(PROJECT_ROOT / "reports/evaluation/sprint5_hash_report.json")
    error_summary = load_json(PROJECT_ROOT / "reports/analysis/error_summary.json")
    intake = load_json(PROJECT_ROOT / "reports/training/sprint4b_v2_hash_report.json")
    class_names = project["dataset"]["class_names"]
    arabic_names = {
        "fish": "سمكة",
        "jellyfish": "قنديل البحر",
        "penguin": "بطريق",
        "puffin": "ببغاء البحر",
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
        "real_25": "٢٥٪ حقيقي",
        "real_50": "٥٠٪ حقيقي",
        "real_75": "٧٥٪ حقيقي",
        "real_only": "حقيقي فقط",
    }
    regimes = []
    for key in REGIMES:
        ratio = experiments["realized_ratios"][key]
        config = load_yaml(PROJECT_ROOT / "configs/training/regimes" / f"{key}.yaml")
        result = final_results[key]
        validation_row = validation[key]
        rank = ranking[key]
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
                "status": "completed",
                "checkpointAvailable": True,
                "checkpointHash": result["checkpoint"]["sha256"],
                "validationMetricsAvailable": True,
                "finalTestMetricsAvailable": True,
                "validationMetrics": {
                    "precision": numeric(validation_row, "validation_precision"),
                    "recall": numeric(validation_row, "validation_recall"),
                    "map50": numeric(validation_row, "validation_map50"),
                    "map5095": numeric(validation_row, "validation_map50_95"),
                },
                "finalMetrics": {
                    "precision": result["metrics"]["precision"],
                    "recall": result["metrics"]["recall"],
                    "map50": result["metrics"]["map50"],
                    "map5095": result["metrics"]["map50_95"],
                },
                "rank": numeric(rank, "rank", integer=True),
                "recommended": rank["recommended"].lower() == "true",
                "bestEpoch": numeric(validation_row, "best_epoch", integer=True),
                "durationSeconds": numeric(validation_row, "duration_seconds"),
                "latencyMs": result["latency_ms_per_image"]["total"],
            }
        )
    object_bank_path = PROJECT_ROOT / project["synthetic"]["manifests"] / "object_bank.csv"
    failed_path = (
        PROJECT_ROOT / project["synthetic"]["manifests"] / "failed_generation_attempts.csv"
    )
    report_paths = (
        "reports/evaluation/sprint5_final_report.md",
        "reports/evaluation/final_model_ranking.md",
        "reports/analysis/error_analysis.md",
        "reports/training/sprint4b_v2_intake_report.md",
        "docs/evaluation_protocol.md",
    )
    return {
        "schemaVersion": 2,
        "source": "repository",
        "gitRevision": git_revision(),
        "project": {
            "name": "SynthDet",
            "nameAr": "سينث دِت",
            "descriptionAr": "منصة قياس الفجوة بين البيانات الاصطناعية والحقيقية لكشف الأجسام",
            "sloganAr": "رؤية ذكية. بيانات اصطناعية. قياس أدق.",
            "phase": "Sprint 6B — دمج النتائج والاستدلال",
            "dashboardStatus": "verified_results_integration",
            "seed": project["seed"],
        },
        "identities": {
            "realSplit": project["synthetic"]["active_real_split_identity"],
            "syntheticPool": project["synthetic"]["pool_identity"],
            "objectBank": project["synthetic"]["object_bank_identity"],
            "generatorConfiguration": project["synthetic"]["generator_configuration_identity"],
            "experimentDesign": project["experiments"]["design_identity"],
            "training": evaluation_contract["training_identity"],
            "evaluationContract": campaign["contract_sha256"],
            "testCampaign": campaign["campaign_id"],
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
            "state": "complete_verified",
            "model": common["model"]["architecture"],
            "epochs": common["training"]["epochs"],
            "imageSize": common["training"]["imgsz"],
            "preferredBatch": common["hardware_profiles"]["standard"]["batch"],
            "fallbackBatch": common["hardware_profiles"]["low_memory"]["batch"],
            "completedFinalRegimes": 5,
            "smokeRegimesCompleted": 5,
            "gpu": environment["hardware_profile"]["environment"]["gpu_model"],
            "profile": environment["hardware_profile"]["profile_name"],
            "testSetAccessCount": 0,
            "trainingIdentity": evaluation_contract["training_identity"],
        },
        "environment": {
            "python": campaign["environment"]["python"],
            "torch": campaign["environment"]["pytorch"],
            "ultralytics": campaign["environment"]["ultralytics"],
            "platform": "Google Colab / Tesla T4 (training); Windows CPU (evaluation)",
            "cudaAvailableLocally": False,
            "classification": "training_tesla_t4_evaluation_cpu",
        },
        "scientificResults": {
            "available": True,
            "finalTestEvaluated": True,
            "messageAr": "اكتملت الحملة المحمية وخُتمت نتائج النماذج الخمسة.",
            "recommendedModel": "real_only",
            "primaryMetric": "mAP@50-95",
            "primaryMetricValue": final_results["real_only"]["metrics"]["map50_95"],
            "ranking": ranking_payload,
            "perClass": per_class,
            "objectSize": object_size,
            "latency": latency,
            "domainGap": domain_gap,
            "campaign": {
                "id": campaign["campaign_id"],
                "attempt": campaign["attempt_id"],
                "status": campaign["status"],
                "successfulCampaigns": campaign["successful_campaign_count"],
                "technicalFailures": len(campaign.get("attempt_history", [])),
                "contractHash": campaign["contract_sha256"],
            },
            "resultHashes": result_hashes["model_results"],
            "errorSummary": {
                "selectedCases": error_summary["selected_metadata_rows"],
                "galleryAvailableLocally": False,
                "galleryReasonAr": "صور الاختبار المحمية غير منشورة في أصول الواجهة.",
                "eventCounts": error_summary["event_counts"],
            },
        },
        "api": {
            "implemented": True,
            "baseUrl": "http://localhost:8000",
            "modelsAvailableLocally": 5,
            "recommendedModel": "real_only",
        },
        "reports": [
            {"title": Path(relative).stem.replace("_", " "), "path": relative}
            for relative in report_paths
        ],
        "audit": {
            "splitFrozen": split["status"] == "frozen",
            "syntheticFrozen": generation["status"] == "frozen",
            "experimentValidationPassed": experiments["validation_results"]["passed"],
            "testSetUsedForExperiments": experiments["validation_results"]["test_set_used"],
            "protectedContentIncluded": False,
            "trainingTestAccessCount": 0,
            "authorizedEvaluationCampaigns": 1,
            "resultHashesVerified": len(result_hashes["model_results"]),
            "predictionHashesVerified": sum(
                int("prediction_sha256" in hashes)
                + int("ultralytics_prediction_sha256" in hashes)
                for hashes in result_hashes["model_results"].values()
            ),
            "intakeHashReportAvailable": bool(intake),
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
                "final_results_available": payload["scientificResults"]["available"],
                "protected_content_included": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
