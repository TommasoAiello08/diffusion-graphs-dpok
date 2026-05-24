# Diffusion Graphs — DPOK + Structured Rewards

Fine-tuning **Stable Diffusion 1.5** with **DPOK** (Diffusion Policy Optimization with KL regularization) using **ImageReward** as the training signal, plus research code for **hierarchical scene-graph rewards** and rich offline evaluation (CLIPScore, VQA, DSG).

Based on [DPOK (Fan et al., NeurIPS 2023)](https://arxiv.org/abs/2305.16381).

## Repository layout

```
├── dpok_imagereward.py      # Main DPOK + ImageReward training (production)
├── eval_report.py           # Generate images + multi-metric evaluation report
├── dpok_eval_metrics.py     # CLIP / ImageReward / VQA / DSG scorers
├── dpok_scene_reward.py     # COCO/PSG/GQA loaders + hierarchical rewards
├── dpok_sg_reward.py        # BLIP scene-graph reward (future DPOK variant)
├── perturb_graph.py         # Scene-graph perturbations for sensitivity tests
├── prompts_*.json           # Paper / holdout / single-prompt lists
├── submit_imagereward*.sh    # SLURM jobs (Bocconi HPC — adapt paths)
├── setup_cluster.sh         # Full HPC training env bootstrap
├── setup_eval_env.sh        # Separate eval env + VQA model cache
├── scripts/
│   ├── build_report_figures.py   # Build figures/ from local CSVs
│   ├── pull_figures_from_hpc.sh  # scp PNGs from cluster
│   └── smoke_test_imports.py
├── src/                     # Earlier research: hierarchical CLIP, FK steering
├── notebooks/               # Scene-graph demos (sg_demo_real.ipynb)
├── data_analysis/           # OpenPSG / GQA exploration notebooks
├── eval-single-prompt/      # Pulled eval tables (job 484798)
├── eval-four-prompts/       # Pulled eval tables + logs (job 484796)
├── figures/                 # Report figures (PNG)
├── additional/              # HANDOFF, PROJECT_DOCS, meeting notes
└── report.txt               # Full project report
```

## Requirements

| Resource | Notes |
|----------|--------|
| **GPU** | Required for training and eval (≥24 GB VRAM recommended; 80 GB for fp32 + large batches) |
| **Disk** | ~15 GB Hugging Face caches (SD1.5, CLIP, ImageReward); +23 GB if using VQA eval |
| **Python** | 3.10 |
| **CUDA** | 12.x (wheels tested with cu121) |

**Not shipped in this repo (download at runtime):**

- `runwayml/stable-diffusion-v1-5` (Hugging Face)
- `ImageReward-v1.0` weights (via `image-reward` package)
- COCO / PSG / GQA datasets (optional; use `prompts_*.json` for paper reproduction)

## Installation

### Option A — pip (local or VM)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
python scripts/smoke_test_imports.py
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate dpok-diffusion-graphs
python scripts/smoke_test_imports.py
```

### Eval environment (separate venv recommended)

VQA/DSG metrics conflict with the training pin set — use a second environment:

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

On Bocconi HPC, use `bash setup_eval_env.sh` (also caches the VQA model).

## Quick start — training (single prompt)

```bash
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

python -u dpok_imagereward.py \
  --prompts_file prompts_single.json \
  --total_samples 500 \
  --sample_batch 4 \
  --grad_steps 5 \
  --is_batch 8 \
  --is_clip 1e-3 \
  --num_steps 20 \
  --pipe_dtype bf16 \
  --lora_rank 4 \
  --guidance_scale 7.5 \
  --alpha 5 \
  --beta 0.01 \
  --lr 1e-4 \
  --grad_norm_clip 1.0 \
  --save_every 50 \
  --save_dir ./dpok_outputs/smoke_run \
  --wandb_project dpok-smoke
```

**Paper-style 4 prompts:**

```bash
python -u dpok_imagereward.py \
  --prompts_file prompts_paper.json \
  --total_samples 10000 \
  --save_dir ./dpok_outputs/paper4
```

**COCO captions** (needs a captions JSON path):

```bash
python -u dpok_imagereward.py \
  --coco /path/to/captions_train2017.json \
  --max_prompts 104 \
  --total_samples 20000 \
  --save_dir ./dpok_outputs/coco104
```

## Evaluation

After training, run (needs GPU + eval deps):

```bash
python -u eval_report.py \
  --lora_path ./dpok_outputs/smoke_run/lora_unet_final \
  --prompts_file prompts_single.json \
  --seeds_per_prompt 10 \
  --save_dir ./dpok_outputs/smoke_run/eval \
  --metrics clip,imagereward,vqa,dsg \
  --run_name "smoke eval"
```

CLIP-only (fast, no `t2v-metrics`):

```bash
python -u eval_report.py \
  --lora_path ./dpok_outputs/smoke_run/lora_unet_final \
  --prompts_file prompts_paper.json \
  --metrics clip,imagereward \
  --save_dir ./eval_out
```

## HPC (Bocconi)

1. Clone repo on login node.
2. `bash setup_cluster.sh` — creates conda env `dpok`, caches SD1.5.
3. `bash setup_eval_env.sh` — creates `dpok-eval`, submits VQA cache job.
4. Edit `submit_imagereward*.sh` — set `WORK`, account, paths.
5. `sbatch submit_imagereward_single.sh` (or `_sh` / `_multi.sh`).
6. Pull results: `bash scripts/pull_figures_from_hpc.sh`

## Figures and report

- Full write-up: [`report.txt`](report.txt)
- Regenerate plots: `python scripts/build_report_figures.py` (see [`figures/README.md`](figures/README.md))

## Research code (`src/`)

Not imported by `dpok_imagereward.py`. Examples:

```bash
cd src && python example_t2i.py          # plain SD1.5
python simple_hier_clip_reward.py        # hierarchical CLIP reward demo
```

Requires `PYTHONPATH=src` or running from `src/`.

## Citation

```bibtex
@inproceedings{fan2023dpok,
  title={DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models},
  author={Fan, Ying and others},
  booktitle={NeurIPS},
  year={2023}
}
```

## License

Code is provided for academic / course use. Stable Diffusion and ImageReward have their own licenses — comply with Hugging Face and ImageReward terms when downloading weights.
