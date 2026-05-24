# Diffusion Graphs — DPOK + Structured Rewards

Fine-tuning **Stable Diffusion 1.5** with **DPOK** (Diffusion Policy Optimization with KL regularization) using **ImageReward** as the training signal, plus research code for **hierarchical scene-graph rewards** and multi-metric evaluation.

Based on [DPOK (Fan et al., NeurIPS 2023)](https://arxiv.org/abs/2305.16381).

---

## Grading checklist (how to navigate this repo)

This repository is organized to match the course requirements:

| Requirement | Where to look |
|-------------|---------------|
| **Readable, documented code** | Module docstrings at the top of `dpok_imagereward.py`, `eval_report.py`, `dpok_eval_metrics.py`, `dpok_scene_reward.py`; function docstrings throughout; research code in `src/` |
| **Clear folder structure** | `src/`, `scripts/`, `notebooks/`, `data/`, `figures/`, `eval-*/` — each folder has a README |
| **Reproducible environment** | `requirements.txt`, `requirements-eval.txt`, `environment.yml`, `setup_cluster.sh` |
| **Install + run experiments** | Sections below: [Installation](#installation), [Experiments](#experiments) |
| **Version control** | Git history on `main` with descriptive commit messages |

---

## Repository layout

```
├── dpok_imagereward.py      # Main DPOK + ImageReward training
├── eval_report.py           # Generate images + multi-metric evaluation report
├── dpok_eval_metrics.py     # CLIP / ImageReward / VQA / DSG scorers
├── dpok_scene_reward.py     # COCO/PSG/GQA loaders + hierarchical rewards
├── dpok_sg_reward.py        # BLIP scene-graph reward (future DPOK variant)
├── perturb_graph.py         # Scene-graph perturbations for sensitivity tests
├── data/
│   └── prompts/             # JSON prompt lists (paper, holdout, single)
├── scripts/                 # smoke tests, figure builder, HPC pull
├── notebooks/               # Scene-graph demos
├── data_analysis/           # OpenPSG / GQA exploration notebooks
├── src/                     # Earlier research: hierarchical CLIP, FK steering
├── eval-single-prompt/      # Pulled eval results (job 484798)
├── eval-four-prompts/       # Pulled eval results (job 484796)
├── figures/                 # Report figures (PNG)
├── additional/              # PROJECT_DOCS, meeting notes
└── report.txt               # Full project report
```

See also: [`data/README.md`](data/README.md), [`scripts/README.md`](scripts/README.md), [`notebooks/README.md`](notebooks/README.md).

---

## Requirements

| Resource | Notes |
|----------|--------|
| **GPU** | Required for training and full eval (≥24 GB VRAM recommended) |
| **Disk** | ~15 GB Hugging Face caches (SD1.5, CLIP, ImageReward); +23 GB for VQA eval |
| **Python** | 3.10 |
| **CUDA** | 12.x (wheels tested with cu121) |

**Downloaded at runtime (not in repo):**

- `runwayml/stable-diffusion-v1-5` (Hugging Face)
- `ImageReward-v1.0` weights (via `image-reward` package)
- COCO / PSG / GQA datasets (optional — use `data/prompts/*.json` for paper reproduction)

---

## Installation

### Option A — pip (recommended)

```bash
git clone https://github.com/TommasoAiello08/diffusion-graphs-dpok.git
cd diffusion-graphs-dpok

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
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

### Eval environment (optional, separate venv)

VQA/DSG metrics use a different dependency set:

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

---

## Experiments

All experiments below use prompt files under `data/prompts/`. Model weights are fetched automatically on first run.

### 1. Smoke test (verify setup, no training)

```bash
python scripts/smoke_test_imports.py
```

### 2. Single-prompt training (paper Sec 5.2 diagnostic)

```bash
export HF_HOME=~/.cache/huggingface

python -u dpok_imagereward.py \
  --prompts_file data/prompts/prompts_single.json \
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

### 3. Four-prompt training (paper Sec 5.2)

```bash
python -u dpok_imagereward.py \
  --prompts_file data/prompts/prompts_paper.json \
  --total_samples 10000 \
  --save_dir ./dpok_outputs/paper4
```

### 4. Evaluation (after training)

```bash
python -u eval_report.py \
  --lora_path ./dpok_outputs/smoke_run/lora_unet_final \
  --prompts_file data/prompts/prompts_single.json \
  --seeds_per_prompt 10 \
  --save_dir ./dpok_outputs/smoke_run/eval \
  --metrics clip,imagereward \
  --run_name "smoke eval"
```

CLIP-only eval is fast; add `vqa,dsg` for full metrics (needs `requirements-eval.txt`).

### 5. Regenerate report figures (no GPU)

```bash
python scripts/build_report_figures.py
```

### 6. Pre-computed results (no GPU needed)

Open the bundled eval folders to inspect completed runs:

- `eval-single-prompt/report.txt` — single-prompt train vs baseline
- `eval-four-prompts/train_prompts/report.txt` — 4-prompt training eval
- `eval-four-prompts/holdout_prompts/report.txt` — holdout generalization
- `figures/` — plots for the written report (`report.txt`)

### 7. HPC (Bocconi cluster)

```bash
bash setup_cluster.sh              # training env + SD1.5 cache
bash setup_eval_env.sh             # eval env + VQA cache
sbatch submit_imagereward_single.sh   # 1-prompt job
sbatch submit_imagereward.sh          # 4-prompt + holdout eval
```

Edit account/paths inside the `submit_*.sh` scripts before running on a different cluster.

---

## Main modules (documentation map)

| Module | Role |
|--------|------|
| `dpok_imagereward.py` | Online DPOK loop: sample trajectories → ImageReward → IS-weighted policy + KL loss → LoRA update |
| `eval_report.py` | Load SD1.5 + LoRA, generate images, score with CLIP/ImageReward/VQA/DSG, write reports |
| `dpok_eval_metrics.py` | Metric scorers and standalone eval on pre-generated image folders |
| `dpok_scene_reward.py` | Dataset loaders (COCO/PSG/GQA) and CLIP-based hierarchical rewards |
| `src/simple_hier_clip_reward.py` | Earlier hierarchical CLIP reward prototype |

Full write-up: [`report.txt`](report.txt). Technical notes: [`additional/PROJECT_DOCS.md`](additional/PROJECT_DOCS.md).

---

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

MIT — see [`LICENSE`](LICENSE). Stable Diffusion and ImageReward have their own licenses; comply with Hugging Face and ImageReward terms when downloading weights.
