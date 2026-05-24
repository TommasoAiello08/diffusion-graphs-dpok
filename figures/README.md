# Report figures

Static PNGs used in the report. Regenerated locally on 2026-05-23 from the CSV / JSON outputs under `results/` via `scripts/build_report_figures.py`.

## Present in this folder

| File | Description |
|------|-------------|
| `pipeline_placeholder.png` | DPOK + ImageReward pipeline diagram |
| `single_per_prompt_curves.png` | ImageReward vs training round — single-prompt fine-tune (job 484798) |
| `four_per_prompt_curves.png` | ImageReward vs training round — 4-prompt fine-tune (job 484796) |
| `single_snapshot_grid.png` | Checkpoint ImageReward scores (text grid; thumbnails pulled separately) |
| `four_snapshot_grid.png` | 4 prompts × training rounds score grid |
| `single_score_distributions.png` | CLIP / IR / VQA / DSG score histograms — single-prompt eval |
| `train_score_distributions.png` | Score distributions — 4 training prompts |
| `holdout_score_distributions.png` | Score distributions — 4 held-out prompts |
| `score_radar_train.png` | Radar chart — training prompts |
| `score_radar_holdout.png` | Radar chart — held-out prompts |
| `score_radar_single.png` | Radar chart — single-prompt eval |
| `per_prompt_delta_bars.png` | Δ ImageReward & Δ VQA — training vs held-out per prompt |

## Missing (need HPC `scp`)

Image-based side-by-side grids and full snapshot grids with thumbnails (the matplotlib-only regeneration cannot recreate the original SD samples):

- `single_baseline_vs_trained_grid.png`
- `four_baseline_vs_trained_grid.png`
- `holdout_baseline_vs_trained_grid.png`

Pull them from the cluster when logged in:

```bash
bash scripts/pull_figures_from_hpc.sh
```

## Regenerate

```bash
export MPLCONFIGDIR="$(pwd)/figures/.mplconfig"
python scripts/build_report_figures.py
```
