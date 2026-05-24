#!/bin/bash
# cache_eval_models.sh
#
# Run ONCE on the LOGIN NODE to download all models needed by dpok-eval.
# After this completes, eval SLURM jobs (eval_quick.sh, submit_experiment.sh)
# find everything locally — no internet required on compute nodes.
#
# Usage:
#   bash cache_eval_models.sh 2>&1 | tee /home/3223837/dpok_outputs/_logs/cache_eval_models.log
#
# Models downloaded (~8-24 GB depending on clip-flant5 variant):
#   bert-base-uncased         — BERT tokenizer + model used by ImageReward
#   clip-flant5-xl            — t2v_metrics VQAScore model (CLIP + FlanT5-XL)
#   (google/flan-t5-xl is fetched automatically as part of clip-flant5-xl)

set -uo pipefail

echo "========================================"
echo "  cache_eval_models.sh"
echo "  Started: $(date)"
echo "========================================"

module purge || true
module load miniconda3 || true
source /software/miniconda3/etc/profile.d/conda.sh

export HF_HOME=/scratch/3223837/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# NO HF_HUB_OFFLINE — this script is the download step
unset HF_HUB_OFFLINE  2>/dev/null || true
unset TRANSFORMERS_OFFLINE 2>/dev/null || true

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

echo ""
echo "HF_HOME : $HF_HOME"
echo "Env     : dpok-eval"
echo ""

conda run -n dpok-eval python - << 'PY'
import os, sys

os.environ["HF_HOME"] = "/scratch/3223837/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/3223837/.cache/huggingface/transformers"

# ── 1. bert-base-uncased (required by ImageReward) ───────────────────────────
print("=" * 60)
print("1/2  bert-base-uncased  (for ImageReward tokenizer)")
print("=" * 60)
from transformers import BertTokenizer, BertModel
tok = BertTokenizer.from_pretrained("bert-base-uncased")
mdl = BertModel.from_pretrained("bert-base-uncased")
print(f"  OK — vocab size {tok.vocab_size}")
del tok, mdl

# ── 2. clip-flant5-xl (required by VQAScore / DSGScore) ──────────────────────
print()
print("=" * 60)
print("2/2  clip-flant5-xl  (VQAScore / DSGScore — may take 20-30 min)")
print("     Downloads: zhiqiulin/clip-flant5-xl + google/flan-t5-xl")
print("     Total size: ~8-24 GB")
print("=" * 60)
import t2v_metrics
scorer = t2v_metrics.VQAScore(model="clip-flant5-xl")
print("  Model loaded.")

# smoke test
from PIL import Image
import tempfile
img = Image.new("RGB", (64, 64), (128, 128, 128))
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    img.save(tmp.name)
    path = tmp.name
result = scorer(images=[path], texts=["a grey square"])
os.unlink(path)
print(f"  Smoke test score: {float(result[0, 0]):.4f}")
del scorer

print()
print("=" * 60)
print("All eval models cached. Ready for offline eval jobs.")
print("=" * 60)
PY

echo ""
echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
