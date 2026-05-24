#!/usr/bin/env python3
"""Build report figures from local CSV/JSON into ./figures/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = [
    ("CLIPScore", "clip"),
    ("ImageReward", "imagereward"),
    ("VQAScore", "vqa"),
    ("DSGScore", "dsg"),
]


def plot_snapshots_curves(csv_path: Path, out_path: Path, title: str) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_prompt: dict[int, list] = {}
    for r in rows:
        p = int(r["prompt_idx"])
        by_prompt.setdefault(p, []).append((int(r["round"]), float(r["ir_score"]), r["prompt_text"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for p_idx in sorted(by_prompt):
        pts = sorted(by_prompt[p_idx], key=lambda x: x[0])
        label = pts[0][2][:45]
        ax.plot([x[0] for x in pts], [x[1] for x in pts], marker="o", label=f"[{p_idx}] {label}")
    ax.set_xlabel("Training round")
    ax.set_ylabel("ImageReward (fixed seed)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_score_distributions(trained_csv: Path, baseline_csv: Path, out_path: Path, title: str) -> None:
    def load(path: Path) -> dict[str, list[float]]:
        data: dict[str, list[float]] = {k: [] for _, k in METRIC_KEYS}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for _, k in METRIC_KEYS:
                    v = row.get(k)
                    if v not in (None, ""):
                        data[k].append(float(v))
        return data

    tr = load(trained_csv)
    bl = load(baseline_csv)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (label, key) in zip(axes, METRIC_KEYS):
        if tr[key]:
            ax.hist(tr[key], bins=15, alpha=0.7, color="steelblue", label=f"Trained μ={np.mean(tr[key]):.3f}")
            ax.axvline(np.mean(tr[key]), color="navy", linestyle="--", lw=1.5)
        if bl[key]:
            ax.hist(bl[key], bins=15, alpha=0.5, color="coral", label=f"Baseline μ={np.mean(bl[key]):.3f}")
            ax.axvline(np.mean(bl[key]), color="darkred", linestyle=":", lw=1.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Score")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_radar_from_report(report_json: Path, out_path: Path, title: str) -> None:
    data = json.loads(report_json.read_text(encoding="utf-8"))
    labels = [l for l, k in METRIC_KEYS]
    keys = [k for _, k in METRIC_KEYS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    def means(block: dict) -> list[float]:
        return [block["aggregate"][k]["mean"] for k in keys]

    tr = means(data["trained"])
    bl = means(data["baseline"])
    tr_plot = tr + [tr[0]]
    bl_plot = bl + [bl[0]]
    ax.plot(angles, tr_plot, "o-", color="steelblue", lw=2, label="Trained")
    ax.fill(angles, tr_plot, color="steelblue", alpha=0.15)
    ax.plot(angles, bl_plot, "s--", color="coral", lw=2, label="Baseline")
    ax.fill(angles, bl_plot, color="coral", alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_delta_bars(train_csv: Path, holdout_csv: Path, out_path: Path) -> None:
    def read_deltas(path: Path) -> tuple[list[str], dict[str, list[float]]]:
        prompts, deltas = [], {"imagereward": [], "vqa": []}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompts.append(row["prompt"][:28])
                deltas["imagereward"].append(float(row["delta_imagereward"]))
                deltas["vqa"].append(float(row["delta_vqa"]))
        return prompts, deltas

    p1, d1 = read_deltas(train_csv)
    p2, d2 = read_deltas(holdout_csv)
    x = np.arange(max(len(p1), len(p2)))
    w = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for col, (name, prompts, d) in enumerate([("Train prompts", p1, d1), ("Holdout prompts", p2, d2)]):
        xi = np.arange(len(prompts))
        for row, metric, color in [(0, "imagereward", "steelblue"), (1, "vqa", "seagreen")]:
            axes[row, col].bar(xi, d[metric], color=color, alpha=0.85)
            axes[row, col].axhline(0, color="black", lw=0.8)
            axes[row, col].set_title(f"{name} — Δ {metric.upper()}")
            axes[row, col].set_xticks(xi)
            axes[row, col].set_xticklabels([p[:22] for p in prompts], rotation=25, ha="right", fontsize=8)
            axes[row, col].grid(axis="y", alpha=0.3)
    fig.suptitle("Per-prompt metric deltas (trained − baseline)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_snapshot_score_grid(csv_path: Path, out_path: Path, title: str) -> None:
    """Text grid of IR scores when image files are unavailable."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    prompts = sorted({int(r["prompt_idx"]) for r in rows})
    rounds = sorted({int(r["round"]) for r in rows})
    lookup = {(int(r["prompt_idx"]), int(r["round"])): float(r["ir_score"]) for r in rows}
    n_p, n_r = len(prompts), len(rounds)
    fig, axes = plt.subplots(n_p, n_r, figsize=(2.2 * n_r, 1.8 * n_p), squeeze=False)
    for i, p in enumerate(prompts):
        for j, rd in enumerate(rounds):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            val = lookup.get((p, rd), float("nan"))
            color = "#d4edda" if val > 0 else "#f8d7da" if val < -0.5 else "#fff3cd"
            ax.set_facecolor(color)
            ax.text(0.5, 0.5, f"r{rd}\n{val:+.2f}", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        axes[i][0].set_ylabel(rows[0]["prompt_text"][:22] if p == 0 else f"P{p}", fontsize=7, rotation=0, ha="right")
    fig.suptitle(title + " (scores only — pull PNGs from HPC for images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, 3.2, "Prompts\n(JSON / COCO)"),
        (2.0, 3.2, "SD1.5 UNet\n+ LoRA (policy)"),
        (4.0, 3.2, "DDIM sample\nCFG=7.5"),
        (6.0, 3.2, "Trajectories\n(x_t, x_{t-1})"),
        (8.0, 3.2, "ImageReward\n(terminal r)"),
        (10.0, 3.2, "IS + KL update\n(ref UNet frozen)"),
        (12.0, 3.2, "LoRA\ncheckpoint"),
        (4.0, 1.0, "Frozen ref UNet\n(KL anchor)"),
        (10.0, 1.0, "eval_report.py\nCLIP/VQA/DSG"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 1.5, 1.0, fill=True, facecolor="#e8f4fc", edgecolor="#333", lw=1.5))
        ax.text(x + 0.75, y + 0.5, text, ha="center", va="center", fontsize=8)
    arrows = [(1.8, 3.7, 2.0, 3.7), (3.5, 3.7, 4.0, 3.7), (5.5, 3.7, 6.0, 3.7), (7.5, 3.7, 8.0, 3.7),
              (9.5, 3.7, 10.0, 3.7), (11.5, 3.7, 12.0, 3.7), (4.75, 2.0, 4.75, 3.2), (10.75, 2.0, 10.75, 3.2)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("DPOK + ImageReward pipeline (Diffusion Graphs)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    plot_pipeline(FIG / "pipeline_placeholder.png")

    plot_snapshots_curves(
        ROOT / "eval-single-prompt/snapshots.csv",
        FIG / "single_per_prompt_curves.png",
        "Single-prompt: ImageReward vs round",
    )
    plot_snapshots_curves(
        ROOT / "eval-four-prompts/snapshots.csv",
        FIG / "four_per_prompt_curves.png",
        "Four-prompt training: ImageReward vs round",
    )
    plot_snapshot_score_grid(
        ROOT / "eval-single-prompt/snapshots.csv",
        FIG / "single_snapshot_grid.png",
        "Single-prompt snapshots",
    )
    plot_snapshot_score_grid(
        ROOT / "eval-four-prompts/snapshots.csv",
        FIG / "four_snapshot_grid.png",
        "Four-prompt snapshots",
    )

    plot_score_distributions(
        ROOT / "eval-single-prompt/trained_scores.csv",
        ROOT / "eval-single-prompt/baseline_scores.csv",
        FIG / "single_score_distributions.png",
        "Single-prompt eval — score distributions",
    )
    plot_score_distributions(
        ROOT / "eval-four-prompts/train_prompts/trained_scores.csv",
        ROOT / "eval-four-prompts/train_prompts/baseline_scores.csv",
        FIG / "train_score_distributions.png",
        "Train prompts — score distributions",
    )
    plot_score_distributions(
        ROOT / "eval-four-prompts/holdout_prompts/trained_scores.csv",
        ROOT / "eval-four-prompts/holdout_prompts/baseline_scores.csv",
        FIG / "holdout_score_distributions.png",
        "Holdout prompts — score distributions",
    )

    plot_radar_from_report(
        ROOT / "eval-four-prompts/train_prompts/report.json",
        FIG / "score_radar_train.png",
        "Metric radar — train prompts",
    )
    plot_radar_from_report(
        ROOT / "eval-four-prompts/holdout_prompts/report.json",
        FIG / "score_radar_holdout.png",
        "Metric radar — holdout prompts",
    )
    plot_radar_from_report(
        ROOT / "eval-single-prompt/report.json",
        FIG / "score_radar_single.png",
        "Metric radar — single prompt",
    )

    plot_delta_bars(
        ROOT / "eval-four-prompts/train_prompts/per_prompt_breakdown.csv",
        ROOT / "eval-four-prompts/holdout_prompts/per_prompt_breakdown.csv",
        FIG / "per_prompt_delta_bars.png",
    )

    # Placeholder note for image grids
    note = FIG / "MISSING_IMAGE_GRIDS_README.txt"
    note.write_text(
        "The following require PNG image folders from HPC (not in git):\n"
        "  - single_baseline_vs_trained_grid.png\n"
        "  - four_baseline_vs_trained_grid.png\n"
        "  - holdout_baseline_vs_trained_grid.png\n"
        "Run: bash scripts/pull_figures_from_hpc.sh\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
