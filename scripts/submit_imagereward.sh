#!/bin/bash
# submit_imagereward.sh
#
# Single ImageReward training run + final eval.
#
# Submit:   sbatch submit_imagereward.sh
# Monitor:  tail -f /home/3223837/dpok_outputs/_logs/imagereward_%j.txt
# Pull:     scp -r 3223837@login.hpc.unibocconi.it:/home/3223837/dpok_outputs/imagereward_test/<JOBID>/ ./
# W&B:      wandb sync /scratch/3223837/.cache/wandb/offline-run-*

#SBATCH --job-name=dpok_imagereward
#SBATCH --account=3223837
#SBATCH --partition=gpunew
#SBATCH --output=/home/3223837/dpok_outputs/_logs/imagereward_%j.txt
#SBATCH --error=/home/3223837/dpok_outputs/_logs/imagereward_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

set -uo pipefail

# Works both as sbatch job and background bash process
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

# Training should stay offline if models are already cached
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

WORK=/home/3223837/dpok-imagereward
cd "$WORK"

TRAIN_PROMPTS="$WORK/prompts/prompts_paper.json"
HOLDOUT_PROMPTS="$WORK/prompts/prompts_holdout.json"
OUT=/home/3223837/dpok_outputs
SAVE_ROOT="$OUT/imagereward_test"
SAVE_DIR="$SAVE_ROOT/$JOB"

# Eval: 30 seeds per prompt × 4 prompts = 120 images per side per pass.
# Matches the paper's Sec 5.3 protocol of "30 images per prompt".
SEEDS_PER_PROMPT=30
METRICS="clip,imagereward,vqa,dsg"

mkdir -p "$OUT/_logs" "$SAVE_ROOT" "$SAVE_DIR"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

nvidia-smi || true

# Eval helper
# Training runs with HF_HUB_OFFLINE=1.
# Eval uses the dpok-eval env and temporarily overrides offline mode.
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
echo "  ImageReward training — $(date)"
echo "════════════════════════════════════════════════════════════════"
echo "Save dir: $SAVE_DIR"
echo ""

TRAIN_STATUS="FAILED"
EVAL_STATUS="SKIPPED"

if python -u src/dpok_imagereward.py \
    --prompts_file "$TRAIN_PROMPTS" \
    --total_samples 10000 \
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
    --wandb_name "imagereward_paper_job${JOB}"; then

    TRAIN_STATUS="OK"

    # ── Eval pass 1: training prompts (paper Sec 5.2 headline number) ─────
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Eval pass 1: TRAINING prompts — $(date)"
    echo "════════════════════════════════════════════════════════════════"
    if eval_run python -u src/eval_report.py \
        --lora_path        "$SAVE_DIR/lora_unet_final" \
        --prompts_file     "$TRAIN_PROMPTS" \
        --seeds_per_prompt $SEEDS_PER_PROMPT \
        --save_dir         "$SAVE_DIR/eval/train_prompts" \
        --metrics          $METRICS \
        --seed             42 \
        --run_name         "DPOK on TRAINING prompts"; then
        EVAL_STATUS="OK (train)"
    else
        EVAL_STATUS="FAILED (train)"
    fi

    # ── Eval pass 2: held-out prompts (generalization check) ──────────────
    # Same 4-axis structure as training (color / multi-object / counting /
    # surreal location), different content. Tells us whether DPOK is
    # learning a transferable composition skill or just memorizing.
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Eval pass 2: HELD-OUT prompts — $(date)"
    echo "════════════════════════════════════════════════════════════════"
    if eval_run python -u src/eval_report.py \
        --lora_path        "$SAVE_DIR/lora_unet_final" \
        --prompts_file     "$HOLDOUT_PROMPTS" \
        --seeds_per_prompt $SEEDS_PER_PROMPT \
        --save_dir         "$SAVE_DIR/eval/holdout_prompts" \
        --metrics          $METRICS \
        --seed             142 \
        --run_name         "DPOK on HELD-OUT prompts"; then
        EVAL_STATUS="$EVAL_STATUS + OK (holdout)"
    else
        EVAL_STATUS="$EVAL_STATUS + FAILED (holdout)"
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
echo "  $SAVE_DIR/eval/train_prompts/report.txt"
echo "  $SAVE_DIR/eval/train_prompts/per_prompt_breakdown.txt"
echo "  $SAVE_DIR/eval/holdout_prompts/report.txt"
echo "  $SAVE_DIR/eval/holdout_prompts/per_prompt_breakdown.txt"
echo "  $SAVE_DIR/snapshot_grid.png  (per-prompt evolution)"
echo "  $SAVE_DIR/per_prompt_curves.png"
echo ""
echo "Next:"
echo "  wandb sync /scratch/3223837/.cache/wandb/offline-run-*"
echo "  scp -r 3223837@login.hpc.unibocconi.it:$SAVE_DIR ./"
echo "========================================"