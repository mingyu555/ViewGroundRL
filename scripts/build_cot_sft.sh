#!/bin/bash
# Build the CoT SFT sets for both datasets, then merge.
#
# Prerequisite: a teacher endpoint is up (scripts/serve_teacher.sh in another shell).
# Everything is resumable — the teacher jsonl caches under $WORK live across runs,
# so an interrupted build picks up where it stopped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW="${RAW:-/mnt/ssd1/vgrl/raw}"
OUT="${OUT:-/mnt/ssd1/vgrl/data}"
WORK="${WORK:-/mnt/ssd1/vgrl/cot}"
NUSC="${NUSC:-/nuscenes}"
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
EXTRA=("${@}")

mkdir -p "$OUT" "$WORK"

echo "### nuScenes planning CoT"
./scripts/dr.sh none python3 data_prep/cot_build.py \
  --task nuscenes \
  --cached_info "$RAW/cached_nuscenes_info.pkl" \
  --split_json create_data/full_split.json --split train \
  --nusc_root "$NUSC" \
  --work_dir "$WORK/nuscenes" \
  --out_json "$OUT/nuscenes_cot_sft.json" \
  "${EXTRA[@]}"

echo "### DriveLM QA CoT"
./scripts/dr.sh none python3 data_prep/cot_build.py \
  --task drivelm \
  --drivelm_json "$RAW/v1_1_train_nus.json" \
  --nusc_root "$NUSC" \
  --work_dir "$WORK/drivelm" \
  --out_json "$OUT/drivelm_cot_sft.json" \
  "${EXTRA[@]}"

echo "### merge"
./scripts/dr.sh none python3 data_prep/cot_merge.py \
  --inputs "$OUT/nuscenes_cot_sft.json" "$OUT/drivelm_cot_sft.json" \
  --out_json "$OUT/cot_sft.json" \
  --val_fraction 0.01

ls -la "$OUT"
