#!/bin/bash
# The publishable experiment set.
#
#   setsid ./scripts/paper_run.sh > /mnt/ssd1/vgrl/logs/paper.log 2>&1 < /dev/null &
#
# Two things changed from the first attempt, both because the first attempt showed
# the method could not be judged from what was measured:
#
# 1. The headline metric is blank-view sensitivity, not nuScenes L2. The RL stage
#    trains on DriveLM QA; L2 scores nuScenes trajectories, so it cannot see what a
#    perception term does. eval_grounding.py asks the finished model the same
#    question three times — all views, evidence view blanked, control view blanked —
#    and reports drop_evidence - drop_control.
#
# 2. kl_clip=1.0 and coef=0.005. Without a cap on the individual KLs they rose
#    together (0.03 -> 169 over 250 steps), leaving the margin as noise.
#
# Ablations, one flag apart, so each claim is separable:
#   off        plain GRPO             — is any change from RL at all?
#   kl1_only   PAPO                   — does the control branch matter?
#   viewground full method
#   literal    +coef*(kl1-kl2)        — wrong-sign negative control; must be worse
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
W=/mnt/ssd1/vgrl; LOGS=$W/logs; SAVES=$W/saves; DATA=$W/data; EVAL=$W/eval
mkdir -p "$EVAL" "$LOGS"
TAG=v2
RL_STEPS=${RL_STEPS:-250}
GROUND_N=${GROUND_N:-800}
step() { echo; echo "######## $(date '+%H:%M:%S')  $*"; }

ground() {   # $1 = model dir, $2 = name
  local out="$EVAL/ground_$2.json"
  [[ -s "$out" ]] && { step "grounding $2 — already measured"; return 0; }
  step "grounding eval: $2"
  ./scripts/dr.sh 4 python3 eval_grounding.py \
      --model "$1" --dataset "$DATA/drivelm_rl_${TAG}_val.json" \
      --out_json "$out" --max_samples "$GROUND_N" --batch_size 64 \
    2>&1 | grep -viE "no platform|kernels|No CUDA|deprecated|it/s\]" | tail -12
}

promote() {  # the final save_model has run out of disk before; recover from a checkpoint
  local d="$1"
  [[ -s "$d/model.safetensors" ]] && return 0
  local ck; ck=$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [[ -n "$ck" && -s "$ck/model.safetensors" ]] || return 1
  step "promoting $(basename "$ck") in $(basename "$d")"
  rm -f "$ck/optimizer.pt"
  for f in model.safetensors config.json generation_config.json chat_template.jinja \
           processor_config.json tokenizer.json tokenizer_config.json; do
    [[ -f "$ck/$f" ]] && mv "$ck/$f" "$d/"
  done
  rm -rf "$ck"
}

# ---- baselines we already have: measure grounding on them ------------------
ground "$SAVES/sft_$TAG"           "sft"            # no RL at all
ground "$SAVES/rl_off_$TAG"        "rl_off"         # plain GRPO
ground "$SAVES/rl_viewground_$TAG" "rl_vg_unclipped"  # the KL-hacked run, kept as evidence

# ---- the fixed runs --------------------------------------------------------
for variant in viewground kl1_only literal; do
  name="rl_${variant}_clip"
  out="$SAVES/$name"
  if promote "$out"; then
    step "RL $variant (clipped) — already trained"
  else
    step "RL $variant (clipped), $RL_STEPS steps"
    extra=()
    [[ "$variant" == "kl1_only" ]] && extra=(view_grounding.sign=kl1_only)
    [[ "$variant" == "literal"  ]] && extra=(view_grounding.sign=literal)
    ./scripts/run_rl.sh_direct \
      model="$SAVES/sft_$TAG" \
      train_file="$DATA/drivelm_rl_${TAG}.json" \
      eval_file="$DATA/drivelm_rl_${TAG}_val.json" \
      output_dir="$out" max_steps="$RL_STEPS" \
      ${extra[@]+"${extra[@]}"} || { echo "!! RL $variant failed"; continue; }
    promote "$out" || { echo "!! no usable checkpoint for $variant"; continue; }
  fi
  ground "$out" "$name"
  # free the optimizer state right away; disk ran out mid-run last time
  rm -f "$out"/checkpoint-*/optimizer.pt 2>/dev/null
  df -h /mnt/ssd1 | tail -1
done

step "PAPER RUN DONE"
echo; echo "=== grounding summary ==="
for f in "$EVAL"/ground_*.json; do
  [[ -s "$f" ]] || continue
  python3 -c "
import json,sys
d=json.load(open('$f'))['summary']
n=d['model'].rstrip('/').split('/')[-1]
for m in ('f1','obj'):
    s=d[m]
    print(f\"  {n:<22} [{m}] n={d['n']:<4} full={s['full']:.3f} \"
          f\"drop_ev={s['drop_evidence']:+.4f} drop_ctrl={s['drop_control']:+.4f} \"
          f\"GAP={s['grounding_gap']:+.4f}\")
"
done
