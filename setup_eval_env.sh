#!/bin/bash
# setup_eval_env.sh
#
# Creates a FRESH dpok-eval conda env (does NOT clone from dpok).
# All packages are pinned to known-compatible versions.
#
# PART 1 — Run on login node (~10-15 min, no GPU needed):
#   bash setup_eval_env.sh
#
# PART 2 — Auto-submits a SLURM job to cache the VQA model on a compute
# node (~30 min, downloads ~23 GB to /scratch cache).
#
# After both parts done, all future sbatch jobs use VQA/DSG automatically.

set -euo pipefail

echo "========================================"
echo "  Setting up dpok-eval (fresh env)"
echo "  Login node: conda + packages"
echo "========================================"

module purge || true
module load miniconda3
module load cuda/12.4
source /software/miniconda3/etc/profile.d/conda.sh

# ── Remove old dpok-eval if exists ───────────────────────────────────────────
conda deactivate 2>/dev/null || true
conda env remove -n dpok-eval -y 2>/dev/null || true

# ── Create fresh Python 3.10 env ─────────────────────────────────────────────
echo "[1/4] Creating fresh dpok-eval env (Python 3.10)..."
conda create -n dpok-eval python=3.10 -y
echo "  Created."

# ── System packages ───────────────────────────────────────────────────────────
echo "[2/4] Installing system packages..."
conda run -n dpok-eval conda install ffmpeg=6.1.2 -c conda-forge -y
echo "  ffmpeg done."

# ── Python packages (pinned compatible set) ───────────────────────────────────
echo "[3/4] Installing Python packages..."

# PyTorch — CUDA 12.1 wheels are forward-compatible with CUDA 12.4
conda run -n dpok-eval pip install \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Core ML packages — pinned to avoid conflicts
conda run -n dpok-eval pip install \
    "numpy<2.0" \
    "huggingface_hub>=0.23,<0.27" \
    "diffusers>=0.27.0,<0.30.0" \
    "transformers>=4.38.0,<4.45.0" \
    "accelerate>=0.27.0,<0.35.0" \
    "peft>=0.9.0,<0.12.0" \
    "safetensors>=0.4.0" \
    "tokenizers>=0.15.0,<0.20.0"

# Eval-specific packages
conda run -n dpok-eval pip install \
    "open_clip_torch>=2.20.0" \
    "matplotlib>=3.7.0" \
    "Pillow>=9.0" \
    "scipy>=1.10.0" \
    "tqdm" \
    "ftfy" \
    "regex" \
    "wandb"

# setuptools needed by image-reward + some t2v-metrics deps
conda run -n dpok-eval pip install setuptools

# OpenAI CLIP from git — required by image-reward; installs as `clip` module
# (different namespace from open_clip_torch `open_clip` → no conflict)
conda run -n dpok-eval pip install "git+https://github.com/openai/CLIP.git"

# ImageReward — human preference reward model (~900 MB weights, cached at runtime)
conda run -n dpok-eval pip install image-reward

# t2v-metrics runtime deps — installed manually with --no-deps to avoid:
#   - image-reward being pulled twice (already above)
#   - hpsv2 / openai / pycocoevalcap (not needed for VQA/DSG)
#   - opencv requiring numpy>=2 (we pin numpy<2.0)
#   - transformers==4.36.1 hard pin (4.44+ is backward compatible for FlanT5)
conda run -n dpok-eval pip install \
    "fire==0.4.0" \
    "gdown>=4.7.1" \
    "iopath" \
    "omegaconf" \
    "opencv-python-headless>=4.6.0,<4.10" \
    "scikit-learn" \
    "sentencepiece>=0.1.99" \
    "tiktoken>=0.7.0" \
    "openai>=1.29.0"

# t2v-metrics itself — --no-deps since all real deps are above
conda run -n dpok-eval pip install --no-deps "t2v-metrics>=1.0,<3.0"

echo "  All packages installed."

# ── Smoke test imports ────────────────────────────────────────────────────────
echo "[4/4] Smoke-testing imports..."

# ── Smoke test ────────────────────────────────────────────────────────────────
conda run -n dpok-eval python - <<'PY'
import torch, diffusers, peft, open_clip, transformers
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  diffusers    {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  peft         {peft.__version__}")
import t2v_metrics
print(f"  t2v_metrics  imported OK")
import ImageReward
print(f"  ImageReward  imported OK")
import clip
print(f"  clip (OpenAI) imported OK")
PY

echo ""
echo "========================================"
echo "  Part 1 done. Submitting Part 2..."
echo "  (VQA model cache job — needs GPU node)"
echo "========================================"

# ── Submit Part 2: cache VQA model on a compute node ─────────────────────────
mkdir -p /home/3223837/dpok_outputs/_logs

CACHE_JOB=$(sbatch --parsable << 'SLURM'
#!/bin/bash
#SBATCH --job-name=cache_eval_models
#SBATCH --account=3223837
#SBATCH --partition=gpu
#SBATCH --output=/home/3223837/dpok_outputs/_logs/cache_eval_%j.txt
#SBATCH --error=/home/3223837/dpok_outputs/_logs/cache_eval_%j.txt
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

set -euo pipefail
echo "=== Caching eval models on compute node === $(date)"

module purge || true
module load miniconda3 cuda/12.4
source /software/miniconda3/etc/profile.d/conda.sh

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/3223837/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# NOT setting HF_HUB_OFFLINE — need internet for download
mkdir -p "$HF_HOME"

conda run -n dpok-eval python - <<'PY'
import os
os.environ["HF_HOME"] = "/scratch/3223837/.cache/huggingface"

import open_clip
print("Caching CLIP ViT-L-14 for CLIPScore...")
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
print("  CLIP ViT-L-14 done.")

from t2v_metrics import VQAScore
print("Downloading + caching CLIP-FlanT5-XXL for VQAScore (~23 GB)...")
print("This takes 20-30 min...")
scorer = VQAScore(model="clip-flant5-xl")
print("  CLIP-FlanT5-XXL cached.")

from PIL import Image
import tempfile, os
img = Image.new("RGB", (64, 64), color=(128, 128, 128))
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
img.save(tmp.name); tmp.close()
score = scorer(images=[tmp.name], texts=["a grey square"])
os.unlink(tmp.name)
print(f"  VQAScore smoke test passed: {float(score[0,0]):.4f}")

import ImageReward as RM
print("Downloading + caching ImageReward model (~900 MB)...")
reward_model = RM.load("ImageReward-v1.0")
print("  ImageReward cached.")
ir_score = reward_model.score("a grey square", [tmp.name + "_ir.png"] )
# just loading is enough — weights are now on disk
print(f"  ImageReward load OK")
print("")
print("All eval models cached. Ready to run VQA/DSG/ImageReward evaluation.")
PY

echo "=== Model caching done === $(date)"
SLURM
)

echo ""
echo "========================================"
echo "  dpok-eval env: READY"
echo "  Model cache job: $CACHE_JOB"
echo ""
echo "  Monitor:  squeue -u 3223837"
echo "  Log:      tail -f /home/3223837/dpok_outputs/_logs/cache_eval_${CACHE_JOB}.txt"
echo ""
echo "  Once cache job finishes (~30 min):"
echo "    sbatch eval_quick.sh          # re-eval job 480154"
echo "    sbatch submit_quick.sh        # new quick run"
echo "    sbatch submit_experiment.sh   # full experiment"
echo "========================================"
