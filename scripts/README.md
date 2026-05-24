# Scripts

Training, evaluation, environment, and utility scripts.

## Training and evaluation (SLURM)

| Script | Purpose |
|--------|---------|
| `submit_imagereward.sh` | 4-prompt DPOK + ImageReward fine-tune followed by training-prompt and held-out evaluation passes. |
| `submit_imagereward_single.sh` | Single-prompt fine-tune (full sample budget on one prompt) + eval. |
| `submit_imagereward_multi.sh` | Multi-prompt fine-tune with many seeds per prompt at eval time. |

All three are SLURM batch scripts targeted at the Bocconi HPC layout. Adjust the `WORK`, `OUT`, and SLURM header lines for a different environment.

## Environment setup

| Script | Purpose |
|--------|---------|
| `setup_cluster.sh` | Build the training conda env (`dpok`) and pre-download Stable Diffusion 1.5 + CLIP weights. |
| `setup_eval_env.sh` | Build the evaluation venv (`dpok-eval`) with `t2v-metrics` for VQAScore / DSGScore / T2I-CompBench. |
| `cache_eval_models.sh` | Pre-download evaluation model weights (CLIP-FlanT5-XXL, etc.). |

## Utilities

| Script | Purpose |
|--------|---------|
| `smoke_test_imports.py` | Verify Python dependencies load and surviving source files compile. Run after `pip install -r requirements.txt`. |
| `build_report_figures.py` | Regenerate PNG figures in `figures/` from CSV / JSON under `results/`. |
| `pull_figures_from_hpc.sh` | `scp` snapshot grids and side-by-side comparison images from Bocconi HPC (requires SSH access). |

## Typical usage

```bash
# Local: install + syntax check
pip install -r requirements.txt
python scripts/smoke_test_imports.py

# Regenerate report figures from results/
python scripts/build_report_figures.py

# Pull missing PNGs from the cluster (edit USER/HOST inside script first)
bash scripts/pull_figures_from_hpc.sh

# Cluster: submit a training job
sbatch scripts/submit_imagereward_single.sh
```
