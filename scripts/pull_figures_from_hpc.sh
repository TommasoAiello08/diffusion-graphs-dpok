#!/bin/bash
# Pull image-based figures from Bocconi HPC into ./figures/
set -uo pipefail

HOST="3223837@login.hpc.unibocconi.it"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIG="$ROOT/figures"
mkdir -p "$FIG"

pull() {
  local remote="$1"
  local local="$2"
  if scp -o ConnectTimeout=15 "$HOST:$remote" "$local" 2>/dev/null; then
    echo "OK  $local"
  else
    echo "FAIL $local  ($remote)"
  fi
}

echo "Pulling training plots (484798 single, 484796 four-prompt)..."

pull /home/3223837/dpok_outputs/imagereward_single/484798/per_prompt_curves.png \
     "$FIG/single_per_prompt_curves_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_single/484798/snapshot_grid.png \
     "$FIG/single_snapshot_grid_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_single/484798/reward_curve.png \
     "$FIG/single_reward_curve.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/per_prompt_curves.png \
     "$FIG/four_per_prompt_curves_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/snapshot_grid.png \
     "$FIG/four_snapshot_grid_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/reward_curve.png \
     "$FIG/four_reward_curve.png"

echo "Pulling eval grids..."

pull /home/3223837/dpok_outputs/imagereward_single/484798/eval/baseline_vs_trained_grid.png \
     "$FIG/single_baseline_vs_trained_grid.png"

pull /home/3223837/dpok_outputs/imagereward_single/484798/eval/score_distributions.png \
     "$FIG/single_score_distributions_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_single/484798/eval/score_radar.png \
     "$FIG/score_radar_single_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/train_prompts/baseline_vs_trained_grid.png \
     "$FIG/four_baseline_vs_trained_grid.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/holdout_prompts/baseline_vs_trained_grid.png \
     "$FIG/holdout_baseline_vs_trained_grid.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/train_prompts/score_distributions.png \
     "$FIG/train_score_distributions_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/holdout_prompts/score_distributions.png \
     "$FIG/holdout_score_distributions_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/train_prompts/score_radar.png \
     "$FIG/score_radar_train_hpc.png"

pull /home/3223837/dpok_outputs/imagereward_test/484796/eval/holdout_prompts/score_radar.png \
     "$FIG/score_radar_holdout_hpc.png"

echo "Done. See $FIG/"
