# Report figures

Generated locally on 2026-05-23 via `scripts/build_report_figures.py`.

## Present in this folder

| File | Description |
|------|-------------|
| `pipeline_placeholder.png` | DPOK + ImageReward pipeline diagram |
| `single_per_prompt_curves.png` | IR vs round — "A green colored rabbit." (job 484798) |
| `four_per_prompt_curves.png` | IR vs round — 4 paper prompts (job 484796) |
| `single_snapshot_grid.png` | Checkpoint IR scores (text grid; no images locally) |
| `four_snapshot_grid.png` | 4 prompts × rounds IR score grid |
| `single_score_distributions.png` | CLIP / IR / VQA / DSG histograms — single-prompt eval |
| `train_score_distributions.png` | Distributions — 4 training prompts |
| `holdout_score_distributions.png` | Distributions — 4 holdout prompts |
| `score_radar_train.png` | Radar — train prompts |
| `score_radar_holdout.png` | Radar — holdout prompts |
| `score_radar_single.png` | Radar — single-prompt eval |
| `per_prompt_delta_bars.png` | Δ ImageReward & Δ VQA — train vs holdout |

## Missing (need HPC `scp`)

Image-based grids from `eval_report.py` and full snapshot grids with thumbnails:

- `single_baseline_vs_trained_grid.png`
- `four_baseline_vs_trained_grid.png`
- `holdout_baseline_vs_trained_grid.png`

Pull when logged into Bocconi:

```bash
bash scripts/pull_figures_from_hpc.sh
```

## Regenerate

```bash
export MPLCONFIGDIR="$(pwd)/figures/.mplconfig"
.venv-figures/bin/python scripts/build_report_figures.py
```
