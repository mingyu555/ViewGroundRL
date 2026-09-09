#!/bin/bash
# Stage 1 via TRL. GPUs fixed to 4-7 (shared box).
#   scripts/run_sft_trl.sh                          # full run
#   scripts/run_sft_trl.sh max_train_rows=2000 num_train_epochs=1   # pilot
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-4,5,6,7}"
N=$(tr ',' '\n' <<<"$GPUS" | grep -c .)
OVR=("${@}")

./scripts/dr.sh "$GPUS" accelerate launch \
  --num_processes "$N" --mixed_precision bf16 \
  train_sft.py --config configs/sft_trl.yaml \
  ${OVR[@]+--overrides "${OVR[@]}"}
