#!/bin/bash
# Stage 2 — view-grounded GRPO on DriveLM.
#   scripts/run_rl.sh                       # our method
#   scripts/run_rl.sh kl1_only              # PAPO baseline (no control branch)
#   scripts/run_rl.sh off                   # plain GRPO baseline
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VARIANT="${1:-viewground}"
GPUS="${GPUS:-4,5,6,7}"

case "$VARIANT" in
  viewground) OVR=() ;;
  kl1_only)   OVR=(view_grounding.sign=kl1_only
                   output_dir=/mnt/ssd1/vgrl/saves/rl_papo) ;;
  literal)    OVR=(view_grounding.sign=literal
                   output_dir=/mnt/ssd1/vgrl/saves/rl_literal) ;;
  off)        OVR=(view_grounding.enabled=false
                   output_dir=/mnt/ssd1/vgrl/saves/rl_plain) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

./scripts/dr.sh "$GPUS" accelerate launch \
  --num_processes "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" \
  train_rl.py --config configs/rl_drivelm.yaml \
  ${OVR[@]+--overrides "${OVR[@]}"}
