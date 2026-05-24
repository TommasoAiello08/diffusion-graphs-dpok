#!/bin/bash
set -euo pipefail

ACCOUNT=3223837
WORK=/home/${ACCOUNT}/simple-hier-clip-reward
CACHE_ROOT=/scratch/${ACCOUNT}/.cache
HF_HOME=${CACHE_ROOT}/huggingface
TRANSFORMERS_CACHE=${HF_HOME}/transformers
HF_DATASETS_CACHE=${CACHE_ROOT}/datasets
WANDB_DIR=${CACHE_ROOT}/wandb
OUT=/home/${ACCOUNT}/dpok_outputs

echo "========================================"
echo "Clean DPOK cluster setup"
echo "========================================"

module purge || true
module load miniconda3
module load cuda/12.4
source /software/miniconda3/etc/profile.d/conda.sh

echo "[1/6] Removing old dpok environment if it exists..."
conda deactivate 2>/dev/null || true
conda env remove -n dpok -y || true
rm -rf "$HOME/.conda/envs/dpok"

echo "[2/6] Cleaning caches and old outputs..."
rm -rf "/home/${ACCOUNT}/.cache/huggingface"
rm -rf "/home/${ACCOUNT}/.cache/datasets"
rm -rf "/home/${ACCOUNT}/.cache/wandb"
rm -rf "/home/${ACCOUNT}/.cache/clip"
rm -rf "/home/${ACCOUNT}/.cache/torch"
rm -rf "${CACHE_ROOT}/huggingface"
rm -rf "${CACHE_ROOT}/datasets"
rm -rf "${CACHE_ROOT}/wandb"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${OUT}"

echo "[3/6] Creating fresh conda environment..."
conda create -n dpok python=3.10 -y
conda activate dpok

echo "[4/6] Installing Python packages..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir \
  huggingface_hub==0.21.4 \
  diffusers==0.27.2 \
  transformers==4.35.2 \
  accelerate==0.25.0 \
  peft==0.7.1 \
  safetensors==0.4.2 \
  datasets==2.18.0 \
  open-clip-torch==2.24.0 \
  wandb==0.16.6 \
  protobuf==4.25.3 \
  tensorboard \
  matplotlib \
  Pillow==10.3.0 \
  numpy==1.26.4 \
  scipy \
  ftfy \
  sentencepiece

export HF_HOME
export TRANSFORMERS_CACHE
export HF_DATASETS_CACHE
export WANDB_DIR
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "[5/6] Verifying imports and versions..."
python - <<'PY'
import sys
import torch
import huggingface_hub
import diffusers
import transformers
import accelerate
import peft
import datasets
import open_clip
import wandb

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("huggingface_hub:", huggingface_hub.__version__)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("datasets:", datasets.__version__)
print("wandb:", wandb.__version__)
print("open_clip: OK")
PY

echo "[6/6] Pre-downloading models and dataset caches..."
python - <<'PY'
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from datasets import load_dataset
import open_clip

print("Caching Stable Diffusion v1.5...")
StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)

print("Caching UNet...")
UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

print("Caching CLIP ViT-B-32 (training reward)...")
open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
print("Caching CLIP ViT-L-14 (eval CLIPScore)...")
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")

print("Caching COCO dataset (fallback)...")
try:
    ds = load_dataset("phiyodr/coco2017", split="train", streaming=False)
    print("COCO rows:", len(ds))
except Exception as e:
    print(f"  (skipped: {e})")

print("All caches ready.")
print("Note: VQA/DSG/T2I models are cached separately via setup_eval_env.sh")
PY

echo "========================================"
echo "Setup complete."
echo "Conda env : dpok"
echo "HF cache  : ${HF_HOME}"
echo "Data cache: ${HF_DATASETS_CACHE}"
echo "W&B dir   : ${WANDB_DIR}"
echo "Outputs   : ${OUT}"
echo "========================================"
