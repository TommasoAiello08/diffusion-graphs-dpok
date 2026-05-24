#!/bin/bash
# submit_imagereward.sh
#
# Single ImageReward training run + final eval.
#
# Submit:   sbatch submit_imagereward.sh
# Monitor:  tail -f /home/3223837/dpok_outputs/_logs/imagereward_%j.txt
# Pull:     scp -r 3223837@login.hpc.unibocconi.it:/home/3223837/dpok_outputs/imagereward_test/<JOBID>/ ./
# W&B:      wandb sync /scratch/3223837/.cache/wandb/offline-run-*

#SBATCH --job-name=dpok_ir_multi
#SBATCH --account=3223837
#SBATCH --partition=gpunew
#SBATCH --output=/home/3223837/dpok_outputs/_logs/imagereward_multi_%j.txt
#SBATCH --error=/home/3223837/dpok_outputs/_logs/imagereward_multi_%j.txt
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

WORK=/home/3223837/simple-hier-clip-reward
cd "$WORK"

# ── Section 5.3 multi-prompt regime ───────────────────────────────────────
# Paper: 104 MS-COCO prompts, value function on, 30 images/prompt eval.
# We sample 104 COCO captions deterministically into a static JSON so that
# training and eval see the exact same prompt set, and so the run is
# reproducible across submissions.
COCO_JSON="$WORK/OpenPSG/data/coco/annotations/captions_train2017.json"
TRAIN_PROMPTS="$WORK/data/prompts/prompts_coco104.json"
OUT=/home/3223837/dpok_outputs
SAVE_ROOT="$OUT/imagereward_multi"
SAVE_DIR="$SAVE_ROOT/$JOB"

SEEDS_PER_PROMPT=30
METRICS="clip,imagereward,vqa,dsg"

mkdir -p "$OUT/_logs" "$SAVE_ROOT" "$SAVE_DIR"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

nvidia-smi || true

# Build the 104-prompt COCO JSON if missing (deterministic, seed=42).
if [ ! -f "$TRAIN_PROMPTS" ]; then
    echo "Building $TRAIN_PROMPTS from COCO ..."
    python -c "
import json, random
random.seed(42)
data = json.load(open('$COCO_JSON'))
caps = sorted({a['caption'].strip() for a in data['annotations']})
random.shuffle(caps)
json.dump(caps[:104], open('$TRAIN_PROMPTS', 'w'), indent=2)
print(f'wrote 104 prompts to $TRAIN_PROMPTS')
"
fi

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

if python -u dpok_imagereward.py \
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
    --wandb_name "imagereward_multi_job${JOB}"; then

    TRAIN_STATUS="OK"

    # Single eval on the 104 training prompts (paper Sec 5.3 protocol:
    # 30 images per prompt = 3120 images per side). With 104 distinct
    # prompts the eval is itself a generalization signal (each prompt
    # only seen ~190 times during training), so no separate holdout pass.
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Eval: 104 training prompts × 30 seeds — $(date)"
    echo "════════════════════════════════════════════════════════════════"
    if eval_run python -u eval_report.py \
        --lora_path        "$SAVE_DIR/lora_unet_final" \
        --prompts_file     "$TRAIN_PROMPTS" \
        --seeds_per_prompt $SEEDS_PER_PROMPT \
        --save_dir         "$SAVE_DIR/eval" \
        --metrics          $METRICS \
        --seed             42 \
        --run_name         "DPOK 104-COCO (paper Sec 5.3)"; then
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
echo "  $SAVE_DIR/eval/per_prompt_breakdown.txt  (104 prompts)"
echo "  $SAVE_DIR/snapshot_grid.png  (per-prompt evolution; 104 rows so big)"
echo "  $SAVE_DIR/per_prompt_curves.png"
echo ""
echo "Next:"
echo "  wandb sync /scratch/3223837/.cache/wandb/offline-run-*"
echo "  scp -r 3223837@login.hpc.unibocconi.it:$SAVE_DIR ./"
echo "========================================"