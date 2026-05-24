#!/usr/bin/env python3
"""
eval_report.py — Publication-ready evaluation report for the DPOK + ImageReward pipeline.

Generates a clean evaluation report including:
  - CLIPScore (ViT-L-14)
  - VQAScore (CLIP-FlanT5-XXL)       — optional, needs t2v-metrics
  - DSGScore (Davidsonian decomposition) — optional, needs VQAScore
  - T2I-CompBench (object/attr/rel)   — optional, needs VQAScore

Outputs under <save_dir>/: report.txt, report.json, report.tex,
per_image_scores.csv, score_distributions.png, score_radar.png.

Usage:
  # CLIPScore only (fast, ~2 min):
  python eval_report.py --lora_path dpok_outputs/lora_unet_final \\
      --prompts_file prompts/prompts_paper.json --save_dir results/eval

  # All metrics (needs ~40GB VRAM):
  python eval_report.py --lora_path dpok_outputs/lora_unet_final \\
      --prompts_file prompts/prompts_paper.json --save_dir results/eval \\
      --metrics clip,vqa,dsg,t2i

  # Baseline only (no LoRA):
  python eval_report.py --prompts_file prompts/prompts_paper.json \\
      --save_dir results/eval/baseline
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import gc

import numpy as np
from PIL import Image

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments for the evaluation report."""
    p = argparse.ArgumentParser(
        description="Evaluation report for the DPOK + ImageReward pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lora_path",   type=str, default=None,
                   help="LoRA weights dir (omit for baseline-only)")
    p.add_argument("--model_id",    type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompts_file", type=str, required=True,
                   help="JSON list of prompts. Each prompt is rendered with "
                        "`--seeds_per_prompt` seeds.")
    p.add_argument("--seeds_per_prompt", type=int, default=10,
                   help="Independent seeds per prompt.")
    p.add_argument("--num_images",  type=int, default=None,
                   help="If set, cap the total number of generated images.")
    p.add_argument("--num_steps",   type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--metrics",     type=str, default="clip",
                   help="Comma-separated: clip,vqa,dsg,t2i")
    p.add_argument("--clip_model",  type=str, default="ViT-L-14")
    p.add_argument("--save_dir",    type=str, default="results/eval")
    p.add_argument("--run_name",    type=str, default=None,
                   help="Name for this run in the report (e.g. 'DPOK LoRA')")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(
    prompts: List[str],
    save_dir: Path,
    model_id: str,
    lora_path: Optional[str],
    num_steps: int,
    guidance_scale: float,
    seed: int,
    label: str = "model",
) -> List[Path]:
    """Load SD1.5 (+ optional LoRA), render ``prompts``, and save PNGs under ``save_dir``."""
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use bf16 to match training dtype — adapter weights are saved in bf16.
    # fp16 would cause a dtype mismatch when merging the adapter and can zero
    # out the LoRA contribution silently.
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\n  Loading SD1.5 ({model_id})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if lora_path:
        print(f"  Loading LoRA from {lora_path}")
        from peft import PeftModel
        peft_unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        # merge_and_unload() folds the LoRA delta directly into the base weights
        # so the modified UNet2DConditionModel is passed to the diffusers pipeline
        # without any PEFT wrapper — guaranteed to apply the adapter.
        pipe.unet = peft_unet.merge_and_unload()
        print(f"  LoRA merged into base UNet")
    pipe.unet.eval()

    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=num_steps,
                       guidance_scale=guidance_scale, generator=gen).images[0]
        p = save_dir / f"{i:04d}.png"
        img.save(p)
        paths.append(p)
        if (i + 1) % 10 == 0 or i == len(prompts) - 1:
            print(f"    [{label}] {i+1}/{len(prompts)} images")

    del pipe
    torch.cuda.empty_cache()
    return paths


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all(
    image_paths: List[Path],
    prompts:     List[str],
    metrics:     List[str],
    clip_model:  str,
    device:      str,
) -> List[Dict]:
    """Score each image–prompt pair with the requested metrics (CLIP, IR, VQA, …)."""
    records = [{"idx": i, "file": p.name, "prompt": prompts[i]}
               for i, p in enumerate(image_paths)]
    images  = [Image.open(p).convert("RGB") for p in image_paths]

    # CLIPScore
    if "clip" in metrics:
        print("  Scoring: CLIPScore...")
        from dpok_eval_metrics import CLIPScorer
        scorer = CLIPScorer(model_name=clip_model, device=device)
        for i, (img, pr) in enumerate(zip(images, prompts)):
            records[i]["clip"] = scorer.score(img, pr)
        del scorer; torch.cuda.empty_cache()

    # ImageReward
    if "imagereward" in metrics:
        try:
            from dpok_eval_metrics import ImageRewardScorer
            print("  Scoring: ImageReward...")
            ir = ImageRewardScorer(device=device)
            for i, (img, pr) in enumerate(zip(images, prompts)):
                records[i]["imagereward"] = ir.score(img, pr)
            del ir; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  WARNING: ImageReward unavailable ({e})")

    # VQAScore
    vqa = None
    if "vqa" in metrics:
        try:
            from dpok_eval_metrics import VQAScorer
            print("  Scoring: VQAScore...")
            vqa = VQAScorer(device=device)
            for i, (img, pr) in enumerate(zip(images, prompts)):
                records[i]["vqa"] = vqa.score(img, pr)
        except Exception as e:
            print(f"  WARNING: VQAScore unavailable ({e})")

    # DSGScore
    if "dsg" in metrics:
        if vqa is None:
            try:
                from dpok_eval_metrics import VQAScorer
                vqa = VQAScorer(device=device)
            except Exception:
                pass
        if vqa:
            from dpok_eval_metrics import DSGScorer
            print("  Scoring: DSGScore...")
            dsg = DSGScorer(vqa)
            for i, (img, pr) in enumerate(zip(images, prompts)):
                s, claims = dsg.score(img, pr)
                records[i]["dsg"]        = s
                records[i]["dsg_claims"] = len(claims) if claims else 0

    # T2I-CompBench
    if "t2i" in metrics:
        if vqa is None:
            try:
                from dpok_eval_metrics import VQAScorer
                vqa = VQAScorer(device=device)
            except Exception:
                pass
        if vqa:
            from dpok_eval_metrics import T2ICompScorer
            print("  Scoring: T2I-CompBench...")
            t2i = T2ICompScorer(vqa)
            for i, (img, pr) in enumerate(zip(images, prompts)):
                r = t2i.score(img, pr)
                records[i]["t2i_obj"]  = r["object"]["mean"]
                records[i]["t2i_attr"] = r["attribute"]["mean"]
                records[i]["t2i_rel"]  = r["relation"]["mean"]
                records[i]["t2i_all"]  = r["overall_mean"]

    if vqa:
        del vqa; torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("CLIPScore",      "clip"),
    ("ImageReward",    "imagereward"),
    ("VQAScore",       "vqa"),
    ("DSGScore",       "dsg"),
    ("T2I-Object",     "t2i_obj"),
    ("T2I-Attribute",  "t2i_attr"),
    ("T2I-Relation",   "t2i_rel"),
    ("T2I-Overall",    "t2i_all"),
]

def agg(records: List[Dict], key: str) -> Optional[Dict]:
    """Aggregate mean/std/median/min/max for metric ``key`` across ``records``."""
    vals = [r[key] for r in records if key in r and r[key] is not None]
    if not vals:
        return None
    return {
        "mean":   round(float(np.mean(vals)), 4),
        "std":    round(float(np.std(vals)),  4),
        "median": round(float(np.median(vals)), 4),
        "min":    round(float(np.min(vals)),  4),
        "max":    round(float(np.max(vals)),  4),
        "n":      len(vals),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    run_name:  str,
    trained:   Optional[List[Dict]],
    baseline:  Optional[List[Dict]],
    metrics:   List[str],
    elapsed:   float,
) -> str:
    """Build a human-readable ASCII table comparing trained vs baseline metrics."""
    lines = []
    W = 78

    lines.append("")
    lines.append("=" * W)
    lines.append(f"  DPOK + ImageReward — EVALUATION REPORT")
    lines.append(f"  Run: {run_name}")
    lines.append("=" * W)

    # ── Summary table ────────────────────────────────────────────────────
    has_baseline = baseline is not None and len(baseline) > 0
    has_trained  = trained  is not None and len(trained)  > 0

    if has_trained and has_baseline:
        lines.append("")
        lines.append(f"  {'Metric':<16}  {'Trained':>8} {'(std)':>7}  "
                     f"{'Baseline':>8} {'(std)':>7}  {'Delta':>8}  {'%':>6}")
        lines.append("  " + "─" * (W - 4))

        for label, key in METRIC_KEYS:
            t = agg(trained, key)
            b = agg(baseline, key)
            if t is None:
                continue

            t_str = f"{t['mean']:.4f}"
            t_std = f"({t['std']:.4f})" if t else ""

            if b:
                b_str = f"{b['mean']:.4f}"
                b_std = f"({b['std']:.4f})"
                delta = t['mean'] - b['mean']
                d_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
                pct   = (delta / b['mean'] * 100) if b['mean'] != 0 else 0.0
                p_str = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
            else:
                b_str = "—"
                b_std = ""
                d_str = "—"
                p_str = "—"

            lines.append(f"  {label:<16}  {t_str:>8} {t_std:>7}  "
                         f"{b_str:>8} {b_std:>7}  {d_str:>8}  {p_str:>6}")

        lines.append("  " + "─" * (W - 4))
        n_t = len(trained)
        n_b = len(baseline)
        lines.append(f"  N images: {n_t} trained, {n_b} baseline")

    elif has_trained:
        lines.append("")
        lines.append(f"  {'Metric':<16}  {'Mean':>8} {'Std':>8} {'Med':>8} {'Min':>8} {'Max':>8}")
        lines.append("  " + "─" * (W - 4))
        for label, key in METRIC_KEYS:
            t = agg(trained, key)
            if t is None:
                continue
            lines.append(f"  {label:<16}  {t['mean']:>8.4f} {t['std']:>8.4f} "
                         f"{t['median']:>8.4f} {t['min']:>8.4f} {t['max']:>8.4f}")
        lines.append("  " + "─" * (W - 4))
        lines.append(f"  N images: {len(trained)}")

    elif has_baseline:
        lines.append("")
        lines.append(f"  {'Metric':<16}  {'Mean':>8} {'Std':>8} {'Med':>8} {'Min':>8} {'Max':>8}")
        lines.append("  " + "─" * (W - 4))
        for label, key in METRIC_KEYS:
            b = agg(baseline, key)
            if b is None:
                continue
            lines.append(f"  {label:<16}  {b['mean']:>8.4f} {b['std']:>8.4f} "
                         f"{b['median']:>8.4f} {b['min']:>8.4f} {b['max']:>8.4f}")
        lines.append("  " + "─" * (W - 4))
        lines.append(f"  N images: {len(baseline)} (baseline)")

    lines.append(f"  Time: {elapsed:.0f}s")
    lines.append("=" * W)
    lines.append("")

    return "\n".join(lines)


def format_latex(
    run_name:  str,
    trained:   Optional[List[Dict]],
    baseline:  Optional[List[Dict]],
) -> str:
    """Generate a LaTeX table fragment."""
    lines = []
    has_baseline = baseline is not None and len(baseline) > 0
    has_trained  = trained  is not None and len(trained)  > 0

    if has_trained and has_baseline:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Evaluation results: " + run_name + r"}")
        lines.append(r"\label{tab:eval}")
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Trained & Std & Baseline & $\Delta$ & Impr. (\%) \\")
        lines.append(r"\midrule")
        for label, key in METRIC_KEYS:
            t = agg(trained, key)
            b = agg(baseline, key)
            if t is None:
                continue
            if b:
                delta = t['mean'] - b['mean']
                pct   = delta / b['mean'] * 100 if b['mean'] != 0 else 0.0
                lines.append(f"{label} & {t['mean']:.4f} & {t['std']:.4f} & "
                             f"{b['mean']:.4f} & {delta:+.4f} & {pct:+.1f}\\% \\\\")
            else:
                lines.append(f"{label} & {t['mean']:.4f} & {t['std']:.4f} & "
                             f"-- & -- & -- \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    elif has_trained or has_baseline:
        recs = trained if has_trained else baseline
        tag  = "Trained" if has_trained else "Baseline"
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + tag + r" scores: " + run_name + r"}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Mean & Std & Min & Max \\")
        lines.append(r"\midrule")
        for label, key in METRIC_KEYS:
            a = agg(recs, key)
            if a is None:
                continue
            lines.append(f"{label} & {a['mean']:.4f} & {a['std']:.4f} & "
                         f"{a['min']:.4f} & {a['max']:.4f} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_distributions(trained, baseline, save_dir):
    """Save overlaid histograms of per-image metric scores to ``score_distributions.png``."""
    recs = trained or baseline
    available = [(l, k) for l, k in METRIC_KEYS
                 if any(k in r and r[k] is not None for r in recs)]
    if not available:
        return

    n   = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, key) in zip(axes, available):
        if trained:
            vals = [r[key] for r in trained if key in r and r[key] is not None]
            if vals:
                ax.hist(vals, bins=15, alpha=0.7, color="steelblue", label=f"Trained (u={np.mean(vals):.3f})")
                ax.axvline(np.mean(vals), color="navy", linestyle="--", lw=1.5)
        if baseline:
            vals = [r[key] for r in baseline if key in r and r[key] is not None]
            if vals:
                ax.hist(vals, bins=15, alpha=0.5, color="coral", label=f"Baseline (u={np.mean(vals):.3f})")
                ax.axvline(np.mean(vals), color="darkred", linestyle=":", lw=1.5)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.suptitle("Score Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: score_distributions.png")


def plot_radar(trained, baseline, save_dir):
    """Radar (spider) chart of all available metrics."""
    recs = trained or baseline
    available = [(l, k) for l, k in METRIC_KEYS
                 if any(k in r and r[k] is not None for r in recs)]
    if len(available) < 3:
        return

    labels = [l for l, _ in available]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    def _vals(records, keys):
        return [np.mean([r[k] for r in records if k in r and r[k] is not None])
                for _, k in keys]

    if trained:
        vals = _vals(trained, available) + [_vals(trained, available)[0]]
        ax.plot(angles, vals, "o-", color="steelblue", lw=2, label="Trained")
        ax.fill(angles, vals, color="steelblue", alpha=0.15)

    if baseline:
        vals = _vals(baseline, available) + [_vals(baseline, available)[0]]
        ax.plot(angles, vals, "s--", color="coral", lw=2, label="Baseline")
        ax.fill(angles, vals, color="coral", alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Metric Radar", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_dir / "score_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: score_radar.png")


def save_csv(records, save_path):
    """Write per-image metric records to a CSV file."""
    if not records:
        return
    keys = list(records[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full evaluation pipeline and write the report under ``--save_dir``."""
    args = parse_args()
    t0   = time.time()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = [m.strip().lower() for m in args.metrics.split(",")]
    baseline_only = args.lora_path is None

    run_name = args.run_name or ("Baseline SD1.5" if baseline_only else
               f"LoRA ({Path(args.lora_path).parent.name})")

    print("=" * 60)
    print(f"  EVAL REPORT — {run_name}")
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Device:  {device}")
    print("=" * 60)

    # ── Load prompts ──────────────────────────────────────────────────────
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        base_prompts = json.load(f)
    if not isinstance(base_prompts, list) or not all(isinstance(p, str) for p in base_prompts):
        raise ValueError(f"--prompts_file must be JSON list of strings: {args.prompts_file}")
    # Repeat each prompt `seeds_per_prompt` times so the eval averages over
    # multiple independent samples per prompt.
    prompts = [p for p in base_prompts for _ in range(args.seeds_per_prompt)]
    if args.num_images is not None:
        prompts = prompts[:args.num_images]
    print(f"  Prompts: {len(prompts)}")

    # Save prompt list
    with open(save_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump({f"{i:04d}.png": p for i, p in enumerate(prompts)}, f, indent=2)

    # ── Generate ALL images first, then score ────────────────────────────
    # Generate everything before loading any scorer — SD1.5 + LoRA and
    # CLIP-FlanT5-XXL (~40 GB) cannot coexist in memory at the same time.
    trained_records  = None
    baseline_records = None

    def flush():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if not baseline_only:
        print(f"\n  Generating {len(prompts)} trained images...")
        t_paths = generate_images(
            prompts, save_dir / "trained", args.model_id, args.lora_path,
            args.num_steps, args.guidance_scale, args.seed, "trained",
        )
        flush()

        print(f"\n  Generating {len(prompts)} baseline images...")
        b_paths = generate_images(
            prompts, save_dir / "baseline", args.model_id, None,
            args.num_steps, args.guidance_scale, args.seed, "baseline",
        )
        flush()

        # SD pipeline is fully unloaded — safe to load large scorers now
        print(f"\n  Scoring trained images...")
        trained_records = score_all(t_paths, prompts, metrics, args.clip_model, device)
        flush()

        print(f"\n  Scoring baseline images...")
        baseline_records = score_all(b_paths, prompts, metrics, args.clip_model, device)
        flush()

    else:
        print(f"\n  Generating {len(prompts)} baseline images...")
        b_paths = generate_images(
            prompts, save_dir / "baseline", args.model_id, None,
            args.num_steps, args.guidance_scale, args.seed, "baseline",
        )
        flush()

        print(f"\n  Scoring baseline images...")
        baseline_records = score_all(b_paths, prompts, metrics, args.clip_model, device)
        flush()

    elapsed = time.time() - t0

    # ── Report ────────────────────────────────────────────────────────────
    report = format_report(run_name, trained_records, baseline_records, metrics, elapsed)
    print(report)

    # Save report.txt
    (save_dir / "report.txt").write_text(report, encoding="utf-8")
    print(f"  Saved: report.txt")

    # Save report.tex
    latex = format_latex(run_name, trained_records, baseline_records)
    (save_dir / "report.tex").write_text(latex, encoding="utf-8")
    print(f"  Saved: report.tex")

    # Save report.json
    report_json = {
        "run_name": run_name,
        "metrics":  metrics,
        "elapsed_sec": round(elapsed, 1),
    }
    if trained_records:
        report_json["trained"] = {
            "per_image": trained_records,
            "aggregate": {k: agg(trained_records, key)
                          for _, key in METRIC_KEYS
                          for k in [key] if agg(trained_records, key)},
        }
    if baseline_records:
        report_json["baseline"] = {
            "per_image": baseline_records,
            "aggregate": {k: agg(baseline_records, key)
                          for _, key in METRIC_KEYS
                          for k in [key] if agg(baseline_records, key)},
        }
    with open(save_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"  Saved: report.json")

    # ── Per-prompt breakdown (only meaningful when prompts are repeated, i.e.
    # --prompts_file mode). Aggregates trained vs baseline scores per unique
    # prompt and writes per_prompt_breakdown.txt + per_prompt_breakdown.csv.
    if args.prompts_file and trained_records and baseline_records:
        unique_prompts = []
        for r in trained_records:
            if r["prompt"] not in unique_prompts:
                unique_prompts.append(r["prompt"])

        def _by_prompt(records, prompt_text, key):
            vals = [r[key] for r in records if r["prompt"] == prompt_text and key in r and r[key] is not None]
            return float(np.mean(vals)) if vals else None

        # Print + write a per-prompt table.
        lines = []
        lines.append("=" * 100)
        lines.append("  PER-PROMPT BREAKDOWN  (mean over `seeds_per_prompt` seeds)")
        lines.append("=" * 100)
        # Header
        keys = [k for _, k in METRIC_KEYS if k in trained_records[0]]
        header = f"  {'Prompt':<55s} " + " ".join(f"{k:>9s}" for k in keys) + "    Δ" + keys[0]
        lines.append(header)
        lines.append("-" * 100)
        for pt in unique_prompts:
            row = f"  {pt[:55]:<55s} "
            deltas = []
            for k in keys:
                t_val = _by_prompt(trained_records, pt, k)
                b_val = _by_prompt(baseline_records, pt, k)
                row += f"{t_val:>+9.4f}" if t_val is not None else f"{'-':>9s}"
                if t_val is not None and b_val is not None:
                    deltas.append((k, t_val - b_val))
                row += " "
            if deltas:
                row += f"   {deltas[0][1]:+.4f}"
            lines.append(row)
        lines.append("-" * 100)
        # Baselines row for reference
        lines.append("  baselines:")
        for pt in unique_prompts:
            row = f"  {pt[:55]:<55s} "
            for k in keys:
                b_val = _by_prompt(baseline_records, pt, k)
                row += f"{b_val:>+9.4f} " if b_val is not None else f"{'-':>9s} "
            lines.append(row)
        lines.append("=" * 100)
        per_prompt_text = "\n".join(lines)
        print("\n" + per_prompt_text)
        (save_dir / "per_prompt_breakdown.txt").write_text(per_prompt_text, encoding="utf-8")

        # CSV form
        with open(save_dir / "per_prompt_breakdown.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prompt"] + [f"trained_{k}" for k in keys] + [f"baseline_{k}" for k in keys] + [f"delta_{k}" for k in keys])
            for pt in unique_prompts:
                t_vals = [_by_prompt(trained_records, pt, k) for k in keys]
                b_vals = [_by_prompt(baseline_records, pt, k) for k in keys]
                d_vals = [(t - b) if (t is not None and b is not None) else None for t, b in zip(t_vals, b_vals)]
                w.writerow([pt] + t_vals + b_vals + d_vals)
        print(f"  Saved: per_prompt_breakdown.txt, per_prompt_breakdown.csv")

        # Side-by-side baseline-vs-trained grid: rows = unique prompts,
        # cols = first seed for baseline + trained.
        try:
            import matplotlib.pyplot as plt
            from PIL import Image as _Image
            n_prompts = len(unique_prompts)
            fig, axes = plt.subplots(n_prompts, 2, figsize=(6, 3.0 * n_prompts), squeeze=False)
            for p_idx, pt in enumerate(unique_prompts):
                b_first = next((r for r in baseline_records if r["prompt"] == pt), None)
                t_first = next((r for r in trained_records if r["prompt"] == pt), None)
                for col, rec, label in [(0, b_first, "baseline"), (1, t_first, "trained")]:
                    ax = axes[p_idx][col]
                    ax.set_xticks([]); ax.set_yticks([])
                    if rec is not None:
                        # File path = save_dir / {label} / {file}
                        fp = save_dir / label / rec["file"]
                        if fp.exists():
                            ax.imshow(_Image.open(fp))
                        ax.set_title(f"{label}", fontsize=9)
                    if col == 0:
                        ax.set_ylabel(pt[:30], fontsize=8, rotation=0, ha="right", va="center")
            comp_path = save_dir / "baseline_vs_trained_grid.png"
            fig.tight_layout()
            fig.savefig(comp_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: baseline_vs_trained_grid.png")
        except Exception as e:
            print(f"  WARNING: side-by-side grid failed: {e}")

    # Save per_image_scores.csv
    if trained_records:
        save_csv(trained_records, save_dir / "trained_scores.csv")
    if baseline_records:
        save_csv(baseline_records, save_dir / "baseline_scores.csv")

    # Plots
    print("\n  Generating plots...")
    plot_distributions(trained_records, baseline_records, save_dir)
    plot_radar(trained_records, baseline_records, save_dir)

    print(f"\n  All outputs in: {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
