#!/bin/bash
# submit_imagereward_single.sh
#
# Paper Sec 5.2 single-prompt fine-tune (Option B / smoke test).
# One model, one prompt, all sample budget concentrated on it.
# This is the regime the paper reports IR 0.84 -> 1.6 in.
#
# Submit:   sbatch submit_imagereward_single.sh
# Monitor:  tail -f /home/3223837/dpok_outputs/_logs/imagereward_single_%j.txt
# Pull:     scp -r 3223837@login.hpc.unibocconi.it:/home/3223837/dpok_outputs/imagereward_single/<JOBID>/ ./
# W&B:      wandb sync /scratch/3223837/.cache/wandb/offline-run-*

#SBATCH --job-name=dpok_ir_single
#SBATCH --account=3223837
#SBATCH --partition=gpunew
#SBATCH --output=/home/3223837/dpok_outputs/_logs/imagereward_single_%j.txt
#SBATCH --error=/home/3223837/dpok_outputs/_logs/imagereward_single_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

set -uo pipefail

JOB=${SLURM_JOB_ID:-manual_$(date +%s)}
NODE=${SLURM_NODELIST:-$(hostname)}

mkdir -p /home/3223837/dpok_outputs/_logs

echo "========================================"
echo "Job ID  : $JOB"
echo "Node    : $NODE"
echo "Started : $(date)"
echo "========================================"

# Environment
module purge || true
module load miniconda3 || true
module load cuda/12.4  || true
source /software/miniconda3/etc/profile.d/conda.sh
conda activate dpok

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

export HF_HOME=/scratch/3223837/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=/scratch/3223837/.cache/datasets
export WANDB_DIR=/scratch/3223837/.cache/wandb
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

WORK=/home/3223837/simple-hier-clip-reward
cd "$WORK"

# ── Paper Sec 5.2: single-prompt regime (STRONGER VARIANT v3) ─────────────
# 1 prompt, all sample budget on it.
#
# History:
#   482909 (paper hyperparams): kL stuck at 3e-4, 50% IS-clipped, no learning
#   483053 (v1 — is_clip↑, alpha↓): IS clipping fixed (5%), but kL STILL flat
#                                    at round 830/2000. Policy not moving.
#
# v2 (proposed but skipped) bumped lr 1e-5→1e-4 and grad_norm_clip 0.1→1.0.
#
# v3 (this run) goes further to address suspected precision/capacity issues:
#   --is_clip         1e-4 → 1e-3   (kept from v1)
#   --alpha           10   → 5      (kept from v1)
#   --lr              1e-5 → 1e-4   (kept from v2)
#   --grad_norm_clip  0.1  → 1.0    (kept from v2)
#   --pipe_dtype      bf16 → fp32   (NEW — bf16 ~3 decimal digits may lose
#                                    LoRA gradient signal in 1e-3 to 1e-4 range)
#   --lora_rank       4    → 16     (NEW — 4× more adapter capacity)
#   --total_samples   20K  → 15K    (fp32+rank16 ~600 samp/h, 15K ≈ 25h)
#
# CFG: dpok_imagereward.py already does CFG in BOTH sampling and policy
# log-prob (lines 472-492 and 557-568). guidance_scale=7.5 is consistent
# across train, policy update, snapshot, and eval. No mismatch to fix.
#
# Caveat: stacking 6 changes at once means we won't know which mattered if
# this works. Strategy is "throw everything at the wall" given limited
# compute budget — bisect later if successful.


TRAIN_PROMPTS="$WORK/prompts_single.json"
OUT=/home/3223837/dpok_outputs
SAVE_ROOT="$OUT/imagereward_single"
SAVE_DIR="$SAVE_ROOT/$JOB"

SEEDS_PER_PROMPT=30
METRICS="clip,imagereward,vqa,dsg"

mkdir -p "$OUT/_logs" "$SAVE_ROOT" "$SAVE_DIR"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

nvidia-smi || true

# Eval helper (dpok-eval env, online for HF model downloads)
eval_run() {
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    conda run --no-capture-output -n dpok-eval \
        env HF_HOME=/scratch/3223837/.cache/huggingface \
            TRANSFORMERS_CACHE=/scratch/3223837/.cache/huggingface/transformers \
            HF_HUB_OFFLINE=0 \
            TRANSFORMERS_OFFLINE=0 \
            PYTHONUNBUFFERED=1 \
            TOKENIZERS_PARALLELISM=false \
        "$@"
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ImageReward training (single prompt) — $(date)"
echo "════════════════════════════════════════════════════════════════"
echo "Save dir: $SAVE_DIR"
echo "Prompt  : $(cat $TRAIN_PROMPTS)"
echo ""

TRAIN_STATUS="FAILED"
EVAL_STATUS="SKIPPED"

if python -u dpok_imagereward.py \
    --prompts_file "$TRAIN_PROMPTS" \
    --total_samples 15000 \
    --sample_batch 10 \
    --grad_steps 5 \
    --is_batch 8 \
    --is_clip 1e-3 \
    --num_steps 20 \
    --pipe_dtype fp32 \
    --lora_rank 4 \
    --guidance_scale 7.5 \
    --alpha 5 \
    --beta 0.01 \
    --lr 1e-4 \
    --grad_norm_clip 1.0 \
    --use_value_function \
    --vf_lr 1e-4 \
    --vf_weight 0.5 \
    --save_every 100 \
    --save_dir "$SAVE_DIR" \
    --wandb_project dpok-experiment \
    --wandb_name "imagereward_single_job${JOB}"; then

    TRAIN_STATUS="OK"

    # Eval on the single training prompt: 30 images, paper protocol.
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Eval: 1 prompt × 30 seeds — $(date)"
    echo "════════════════════════════════════════════════════════════════"
    if eval_run python -u eval_report.py \
        --lora_path        "$SAVE_DIR/lora_unet_final" \
        --prompts_file     "$TRAIN_PROMPTS" \
        --seeds_per_prompt $SEEDS_PER_PROMPT \
        --save_dir         "$SAVE_DIR/eval" \
        --metrics          $METRICS \
        --seed             42 \
        --run_name         "DPOK single-prompt (paper Sec 5.2)"; then
        EVAL_STATUS="OK"
    else
        EVAL_STATUS="FAILED"
    fi
else
    TRAIN_STATUS="FAILED"
    EVAL_STATUS="SKIPPED (training failed)"
fi

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "Train    : $TRAIN_STATUS"
echo "Eval     : $EVAL_STATUS"
echo "Outputs  : $SAVE_DIR"
echo "  $SAVE_DIR/eval/report.txt"
echo "  $SAVE_DIR/eval/per_prompt_breakdown.txt"
echo "  $SAVE_DIR/snapshot_grid.png"
echo "  $SAVE_DIR/per_prompt_curves.png"
echo ""
echo "Next:"
echo "  wandb sync /scratch/3223837/.cache/wandb/offline-run-*"
echo "  scp -r 3223837@login.hpc.unibocconi.it:$SAVE_DIR ./"
echo "========================================"
