#!/bin/bash
# End-to-end: CoT labels -> SFT -> eval -> view-grounded RL -> eval.
#
# Launch detached so it survives SSH drops and agent-session teardown:
#   setsid nohup ./scripts/full_run.sh > /mnt/ssd1/vgrl/logs/full_run.log 2>&1 < /dev/null &
# Check progress with scripts/status.sh
#
# Every stage is resumable: teacher output is cached per uid, and each stage skips
# if its output already exists. Re-running after an interruption picks up where it
# stopped rather than redoing hours of teacher calls.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD

W=/mnt/ssd1/vgrl
RAW=$W/raw
DATA=$W/data
LOGS=$W/logs
SAVES=$W/saves
REPRO=/mnt/ssd1/minddriver_repro
mkdir -p "$DATA" "$LOGS" "$SAVES" "$W/cot"

N_NUSC=${N_NUSC:-6000}
N_DRIVELM_QA=${N_DRIVELM_QA:-"perception=1,prediction=1,planning=1,behavior=1"}
SFT_EPOCHS=${SFT_EPOCHS:-8}
RL_STEPS=${RL_STEPS:-250}
TAG=${TAG:-v2}

TEACHER_URLS='http://172.17.0.1:8004/v1,http://172.17.0.1:8005/v1,http://172.17.0.1:8006/v1,http://172.17.0.1:8007/v1'

step() { echo; echo "######## $(date '+%H:%M:%S')  $*"; }
die() { echo "!!!! FAILED: $*"; exit 1; }

# ---------------------------------------------------------------- 1. CoT labels
if [[ -s "$DATA/cot_${TAG}.json" ]]; then
  step "1. CoT labels — already built, skipping"
else
  step "1a. start teacher servers (72B AWQ, TP=1 x 4)"
  ./scripts/serve_teacher_dp.sh || die "teacher servers"

  step "1b. nuScenes CoT (n=$N_NUSC, with ego status, fut_valid_flag filter)"
  ./scripts/dr.sh none env TEACHER_BASE_URL="$TEACHER_URLS" TEACHER_MODEL=teacher-72b \
    python3 data_prep/cot_build.py --task nuscenes \
      --cached_info "$RAW/cached_nuscenes_info.pkl" \
      --split_json create_data/full_split.json --split train \
      --nusc_root /nuscenes --ego_status_json "$W/cache/ego_status.json" \
      --work_dir "$W/cot/nusc_$TAG" --out_json "$DATA/nusc_cot_$TAG.json" \
      --max_samples "$N_NUSC" --concurrency 12 \
      --model teacher-72b --base_url "$TEACHER_URLS" || die "nuScenes CoT"

  step "1c. DriveLM CoT (hint-first: pass 1 only accepted 5.5%, so skip it)"
  ./scripts/dr.sh none env TEACHER_BASE_URL="$TEACHER_URLS" TEACHER_MODEL=teacher-72b \
    python3 data_prep/cot_build.py --task drivelm \
      --drivelm_json "$RAW/v1_1_train_nus.json" --nusc_root /nuscenes \
      --work_dir "$W/cot/dlm_$TAG" --out_json "$DATA/dlm_cot_$TAG.json" \
      --qa_per_frame "$N_DRIVELM_QA" --concurrency 12 \
      --model teacher-72b --base_url "$TEACHER_URLS" || die "DriveLM CoT"

  step "1d. merge (frame-level split)"
  ./scripts/dr.sh none python3 data_prep/cot_merge.py \
    --inputs "$DATA/nusc_cot_$TAG.json" "$DATA/dlm_cot_$TAG.json" \
    --out_json "$DATA/cot_$TAG.json" --val_fraction 0.02 || die "merge"

  step "1e. stop teacher servers (free the GPUs for training)"
  ./scripts/serve_teacher_dp.sh stop
fi

# ---------------------------------------------------------------- 2. SFT
if [[ -s "$SAVES/sft_$TAG/model.safetensors" ]]; then
  step "2. SFT — already trained, skipping"
else
  step "2. SFT ($SFT_EPOCHS epochs)"
  ./scripts/run_sft_trl.sh \
    train_file="$DATA/cot_$TAG.json" eval_file="$DATA/cot_${TAG}_val.json" \
    output_dir="$SAVES/sft_$TAG" num_train_epochs="$SFT_EPOCHS" \
    save_steps=500 eval_steps=250 || die "SFT"
fi

# ------------------------------------------------- 3. val prompts + SFT eval
step "3a. build val prompts (same ego status the SFT prompts carried)"
./scripts/dr.sh none python3 data_prep/nuscenes_val_prompts.py \
  --cached_info "$RAW/cached_nuscenes_info.pkl" \
  --split_json create_data/full_split.json --split val \
  --nusc_root /nuscenes --ego_status_json "$W/cache/ego_status.json" \
  --out_json "$DATA/nusc_val_prompts_$TAG.json" --check_images || die "val prompts"
cp "$DATA/nusc_val_prompts_$TAG.json" "$REPRO/work/nusc_val_prompts_$TAG.json"

eval_model() {   # $1 = model dir, $2 = run name
  local model="$1" name="$2"
  step "eval $name — inference"
  rm -rf "$REPRO/out/$name"
  (cd "$REPRO" && ./work/infer_sft.sh "$model" \
      "/mnt/ssd1/vgrl/data/nusc_val_prompts_$TAG.json" "out/$name" 0 192) || return 1
  step "eval $name — scoring"
  (cd "$REPRO" && ./work/dr.sh none python3 /repro/work/build_pred_base.py \
      --pred_dir "/repro/out/$name" --val_json "/repro/work/nusc_val_prompts_$TAG.json" \
      --out_prefix "/repro/out/$name/eval_traj") || return 1
  for m in uniad stp3; do
    (cd "$REPRO" && ./work/dr.sh none bash -c \
      "cd /repro/work/eval && MPLBACKEND=Agg python3 evaluation.py --metric $m \
         --result_file /repro/out/$name/eval_traj_strict.json --method $name") \
      > "$LOGS/eval_${name}_${m}.log" 2>&1
    echo "---- $name / $m ----"
    tr '\r' '\n' < "$LOGS/eval_${name}_${m}.log" \
      | grep -vE "^\s*[0-9]+%\||no platform|^\s*$|it/s\]" | tail -3
  done
}

eval_model "$SAVES/sft_$TAG" "sft_$TAG" || die "SFT eval"

# ---------------------------------------------------------------- 4. RL
step "4. build RL dataset from DriveLM (evidence views for the perception term)"
./scripts/dr.sh none python3 data_prep/drivelm_prepare.py \
  --drivelm_json "$RAW/v1_1_train_nus.json" --nusc_root /nuscenes \
  --out_rl "$DATA/drivelm_rl_$TAG.json" --check_images \
  --require_evidence_for_rl --val_fraction 0.02 || die "RL dataset"

for variant in viewground off; do
  out="$SAVES/rl_${variant}_$TAG"
  if [[ -s "$out/model.safetensors" ]]; then
    step "4. RL $variant — already trained, skipping"
  else
    step "4. RL $variant ($RL_STEPS steps)"
    args=(model="$SAVES/sft_$TAG"
          train_file="$DATA/drivelm_rl_$TAG.json"
          eval_file="$DATA/drivelm_rl_${TAG}_val.json"
          output_dir="$out" max_steps="$RL_STEPS")
    [[ "$variant" == "off" ]] && args+=(view_grounding.enabled=false)
    ./scripts/run_rl.sh_direct "${args[@]}" || { echo "!! RL $variant failed"; continue; }
  fi
  eval_model "$out" "rl_${variant}_$TAG" || echo "!! eval rl_$variant failed"
done

step "ALL DONE"
