# DPOK + ImageReward — Fine-Tuning Stable Diffusion 1.5

Implementation of an online RL fine-tuning pipeline for **Stable Diffusion 1.5** built around **DPOK** (Diffusion Policy Optimization with KL regularization) and **ImageReward** as the learned reward signal. The policy is a **LoRA** adapter on the UNet attention projections; updates use **PPO-style importance sampling** and a **KL anchor** to a frozen reference UNet. Evaluation is multi-metric (CLIPScore, ImageReward, VQAScore, DSGScore).

The accompanying report describes the method, hyperparameters, and results in full.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
│
├── src/
│   ├── dpok_imagereward.py     # training loop (DPOK + ImageReward + LoRA)
│   ├── eval_report.py          # multi-metric evaluation report
│   └── dpok_eval_metrics.py    # CLIPScore, VQAScore, DSGScore, T2I-CompBench
│
├── prompts/
│   ├── prompts_paper.json      # 4 compositional training prompts
│   ├── prompts_holdout.json    # 4 held-out generalization prompts
│   └── prompts_single.json     # 1 prompt for the single-prompt regime
│
├── scripts/
│   ├── submit_imagereward.sh           # SLURM: 4-prompt train + train/holdout eval
│   ├── submit_imagereward_single.sh    # SLURM: single-prompt fine-tune
│   ├── submit_imagereward_multi.sh     # SLURM: multi-prompt fine-tune
│   ├── setup_cluster.sh                # build the training conda env
│   ├── setup_eval_env.sh               # build the evaluation venv (t2v-metrics)
│   ├── cache_eval_models.sh            # pre-download eval model weights
│   ├── build_report_figures.py         # regenerate report figures from results/
│   ├── smoke_test_imports.py           # syntax / import sanity check
│   └── pull_figures_from_hpc.sh        # scp helper for HPC artifacts
│
├── figures/                    # report figures (PNG)
│
└── results/
    ├── single_prompt/          # single-prompt fine-tune + eval artifacts
    └── four_prompts/           # 4-prompt fine-tune + train/holdout eval artifacts
```

## Installation

Two environments are used in practice: one for training and one for evaluation. They are pinned to different versions because `t2v-metrics` (VQAScore / DSGScore) needs an older PyTorch / transformers combination than the training stack.

### Training environment

```bash
conda env create -f environment.yml
conda activate dpok-imagereward
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Evaluation environment (optional, for VQA / DSG / T2I)

```bash
bash scripts/setup_eval_env.sh     # creates .venv-eval and installs t2v-metrics
bash scripts/cache_eval_models.sh  # pre-downloads CLIP-FlanT5-XXL and friends
```

The training scripts switch between the two envs automatically via `conda run -n dpok-eval`.

## Reproducing the experiments

All experiments run from `prompts/*.json` prompt lists.

### Single-prompt fine-tune (Sec. 5.2 of the report)

```bash
python -u src/dpok_imagereward.py \
    --prompts_file prompts/prompts_single.json \
    --total_samples 15000 --sample_batch 10 --grad_steps 5 \
    --is_batch 8 --is_clip 1e-3 --pipe_dtype fp32 \
    --lora_rank 4 --guidance_scale 7.5 \
    --alpha 5 --beta 0.01 --lr 1e-4 --grad_norm_clip 1.0 \
    --use_value_function --vf_lr 1e-4 --vf_weight 0.5 \
    --save_dir dpok_outputs/single
```

### Four-prompt fine-tune + held-out evaluation (Sec. 5.3)

```bash
python -u src/dpok_imagereward.py \
    --prompts_file prompts/prompts_paper.json \
    --total_samples 10000 --sample_batch 10 --grad_steps 5 \
    --is_batch 8 --is_clip 1e-3 --pipe_dtype fp32 \
    --use_value_function --save_dir dpok_outputs/four
```

### Evaluation (after training)

```bash
python -m src.eval_report \
    --lora_path dpok_outputs/four/lora_unet_final \
    --prompts_file prompts/prompts_paper.json \
    --seeds_per_prompt 30 \
    --metrics clip,imagereward,vqa,dsg \
    --save_dir results/four_prompts/train_prompts

python -m src.eval_report \
    --lora_path dpok_outputs/four/lora_unet_final \
    --prompts_file prompts/prompts_holdout.json \
    --seeds_per_prompt 30 \
    --metrics clip,imagereward,vqa,dsg \
    --save_dir results/four_prompts/holdout_prompts
```

### SLURM (Bocconi HPC)

The full pipelines used in the report are wired up under `scripts/`:

```bash
sbatch scripts/submit_imagereward_single.sh   # single-prompt regime
sbatch scripts/submit_imagereward.sh          # 4-prompt regime + holdout eval
sbatch scripts/submit_imagereward_multi.sh    # multi-prompt regime
```

Paths in the SLURM scripts assume `WORK=/home/<account>/dpok-imagereward`; edit `WORK` and SLURM headers to match your account.

## Results

Pre-computed evaluation tables, snapshots and plots from the report's HPC runs are in:

- `results/single_prompt/` — single-prompt fine-tune (job 484798)
- `results/four_prompts/` — 4-prompt fine-tune (job 484796) with both training and held-out evaluation under `train_prompts/` and `holdout_prompts/`

Each subfolder contains `report.txt`, `report.json`, `report.tex`, per-image / per-prompt CSVs, and PNG plots.

## Figures

Report figures live under `figures/` and can be regenerated from `results/` with:

```bash
python scripts/build_report_figures.py
```

## License

MIT — see `LICENSE`.
