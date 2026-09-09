#!/bin/bash
# Stage 0 — build the SFT and RL datasets.
#
# Needs, under /mnt/ssd1/vgrl/raw/:
#   v1_1_train_nus.json        DriveLM QA  (hf: OpenDriveLab/DriveLM)
#   cached_nuscenes_info.pkl   MindDriver's cached ego/traj info
#   full_split.json            nuScenes train/val token split (in create_data/)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW=/mnt/ssd1/vgrl/raw
OUT=/mnt/ssd1/vgrl/data
NUSC=/nuscenes
mkdir -p "$OUT"

echo "### DriveLM -> SFT + RL"
./scripts/dr.sh none python3 data_prep/drivelm_prepare.py \
  --drivelm_json "$RAW/v1_1_train_nus.json" \
  --nusc_root "$NUSC" \
  --out_sft "$OUT/drivelm_sft.json" \
  --out_rl  "$OUT/drivelm_rl.json" \
  --answer_view_source both \
  --check_images

echo "### nuScenes planning -> SFT"
./scripts/dr.sh none python3 data_prep/nuscenes_sft.py \
  --cached_info "$RAW/cached_nuscenes_info.pkl" \
  --split_json create_data/full_split.json \
  --split train \
  --nusc_root "$NUSC" \
  --out_json "$OUT/nuscenes_plan_sft.json" \
  --check_images

echo "### done"; ls -la "$OUT"
