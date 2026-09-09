#!/bin/bash
# What's left after the view-grounded RL run finished: the plain-GRPO control and
# the evaluation of both RL models.
#
#   setsid ./scripts/finish_run.sh > /mnt/ssd1/vgrl/logs/finish.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
W=/mnt/ssd1/vgrl; LOGS=$W/logs; SAVES=$W/saves; DATA=$W/data
REPRO=/mnt/ssd1/minddriver_repro; TAG=v2
step() { echo; echo "######## $(date '+%H:%M:%S')  $*"; }

eval_model() {
  local model="$1" name="$2"
  [[ -s "$REPRO/out/$name/eval_traj_strict.json" ]] && { step "eval $name — already done"; return 0; }
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

# 1. evaluate the view-grounded model we already trained
eval_model "$SAVES/rl_viewground_$TAG" "rl_viewground_$TAG" || echo "!! eval viewground failed"

# 2. plain-GRPO control, so the term's contribution is separable from RL itself
if [[ -s "$SAVES/rl_off_$TAG/model.safetensors" ]]; then
  step "RL off — already trained"
else
  step "RL off (250 steps, plain GRPO control)"
  ./scripts/run_rl.sh_direct \
    model="$SAVES/sft_$TAG" \
    train_file="$DATA/drivelm_rl_${TAG}.json" \
    eval_file="$DATA/drivelm_rl_${TAG}_val.json" \
    output_dir="$SAVES/rl_off_$TAG" max_steps=250 \
    view_grounding.enabled=false || echo "!! RL off failed"
  # the final save_model ran out of disk last time; promote the checkpoint if needed
  if [[ ! -s "$SAVES/rl_off_$TAG/model.safetensors" ]]; then
    ck=$(ls -d "$SAVES/rl_off_$TAG"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [[ -n "$ck" && -s "$ck/model.safetensors" ]]; then
      step "promoting $(basename "$ck") (final save_model did not complete)"
      rm -f "$ck/optimizer.pt"
      for f in model.safetensors config.json generation_config.json chat_template.jinja \
               processor_config.json tokenizer.json tokenizer_config.json; do
        [[ -f "$ck/$f" ]] && mv "$ck/$f" "$SAVES/rl_off_$TAG/"
      done
      rm -rf "$ck"
    fi
  fi
fi

eval_model "$SAVES/rl_off_$TAG" "rl_off_$TAG" || echo "!! eval off failed"
step "FINISH DONE"
