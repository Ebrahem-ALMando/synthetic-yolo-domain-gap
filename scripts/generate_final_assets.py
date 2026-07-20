"""Generate publication figures and tables only from sealed Sprint 5 CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "reports" / "evaluation"
FIGURES = ROOT / "reports" / "final" / "figures"
TABLES = ROOT / "reports" / "final" / "tables"
REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")
LABELS = ("Synthetic only", "25% real", "50% real", "75% real", "Real only")
COLORS = ("#64748b", "#38bdf8", "#14b8a6", "#8b5cf6", "#2563eb")


def read_csv(name: str) -> list[dict[str, str]]:
    with (EVALUATION / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)


def final_metrics() -> None:
    rows = read_csv("final_test_metrics.csv")
    ranking = read_csv("final_model_ranking.csv")
    by_id = {row["regime"]: row for row in rows}
    ordered = [by_id[regime] for regime in REGIMES]
    metrics = ("precision", "recall", "map50", "map50_95")
    names = ("Precision", "Recall", "mAP@50", "mAP@50-95")
    x = np.arange(len(REGIMES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for index, (metric, label) in enumerate(zip(metrics, names, strict=True)):
        ax.bar(
            x + (index - 1.5) * width, [float(row[metric]) for row in ordered], width, label=label
        )
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Score")
    ax.set_title("Final protected-test comparison (68 real images)")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_01_final_metrics.png", bbox_inches="tight")
    plt.close(fig)
    write_csv(
        "table_01_final_metrics.csv",
        [
            {
                key: row[key]
                for key in ("rank", "regime", "precision", "recall", "map50", "map50_95")
            }
            for row in ranking
        ],
        ["rank", "regime", "precision", "recall", "map50", "map50_95"],
    )


def domain_gap() -> None:
    rows = sorted(
        read_csv("domain_gap_analysis.csv"), key=lambda row: float(row["real_percentage"])
    )
    x = [float(row["real_percentage"]) for row in rows]
    y = [float(row["map50_95"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, y, color="#2563eb", linewidth=2.6, marker="o", markersize=8)
    for x_value, y_value in zip(x, y, strict=True):
        ax.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("Real images in training regime (%)")
    ax.set_ylabel("mAP@50-95 on fixed real test")
    ax.set_ylim(0.15, 0.23)
    ax.set_title("Observed synthetic-to-real domain-gap curve")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_02_domain_gap.png", bbox_inches="tight")
    plt.close(fig)


def per_class() -> None:
    rows = read_csv("per_class_metrics.csv")
    matrix = np.zeros((len(REGIMES), 7), dtype=float)
    class_names = [""] * 7
    for row in rows:
        matrix[REGIMES.index(row["regime"]), int(row["class_id"])] = float(row["ap50_95"])
        class_names[int(row["class_id"])] = row["class_name"]
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.5, float(matrix.max())))
    ax.set_xticks(range(7), class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(REGIMES)), LABELS)
    ax.set_title("Per-class AP@50-95 on the protected real test")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="AP@50-95", fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_03_per_class_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    winners: list[dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        column = matrix[:, class_id]
        winner_index = int(np.argmax(column))
        winners.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "winning_regime": REGIMES[winner_index],
                "ap50_95": f"{column[winner_index]:.6f}",
            }
        )
    write_csv(
        "table_02_per_class_winners.csv",
        winners,
        ["class_id", "class_name", "winning_regime", "ap50_95"],
    )


def object_sizes() -> None:
    rows = read_csv("object_size_metrics.csv")
    sizes = ("small", "medium", "large")
    by_key = {(row["regime"], row["size"]): row for row in rows}
    x = np.arange(len(sizes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for index, regime in enumerate(REGIMES):
        values = [float(by_key[(regime, size)]["map50_95"]) for size in sizes]
        ax.bar(x + (index - 2) * width, values, width, color=COLORS[index], label=LABELS[index])
    ax.set_xticks(x, [value.title() for value in sizes])
    ax.set_ylabel("Descriptive AP@50-95")
    ax.set_title("Performance under frozen original-pixel object-size strata")
    ax.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_04_object_sizes.png", bbox_inches="tight")
    plt.close(fig)
    winners = []
    for size in sizes:
        winner = max(
            (by_key[(regime, size)] for regime in REGIMES), key=lambda row: float(row["map50_95"])
        )
        winners.append(
            {
                "size": size,
                "winning_regime": winner["regime"],
                "ground_truth_instances": winner["ground_truth_instances"],
                "map50_95": winner["map50_95"],
            }
        )
    write_csv(
        "table_03_object_size_winners.csv",
        winners,
        ["size", "winning_regime", "ground_truth_instances", "map50_95"],
    )


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    stages = (
        "Leakage-safe\nSplit V2",
        "Train-only\ncopy-paste pool",
        "Five fixed\nYOLO11n regimes",
        "Frozen Sprint 5\ncontract",
        "One sealed\n68-image campaign",
        "Dashboard +\nFastAPI",
    )
    for index, stage in enumerate(stages):
        left = 0.02 + index * 0.16
        ax.text(
            left + 0.065,
            0.5,
            stage,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": COLORS[index % len(COLORS)],
                "edgecolor": "none",
            },
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(left + 0.155, 0.5),
                xytext=(left + 0.137, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 2},
            )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_05_workflow.png", bbox_inches="tight", transparent=False)
    plt.close(fig)


def main() -> int:
    configure()
    final_metrics()
    domain_gap()
    per_class()
    object_sizes()
    workflow()
    print(f"Generated final assets in {FIGURES.relative_to(ROOT)} and {TABLES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
