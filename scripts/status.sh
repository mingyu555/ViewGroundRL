#!/bin/bash
# Progress of the detached full_run.sh. Safe to call any time.
W=/mnt/ssd1/vgrl
REPRO=/mnt/ssd1/minddriver_repro
TAG=${TAG:-v2}

echo "=== 실행 중? ==="
pgrep -f "full_run.sh" >/dev/null && echo "  full_run.sh 진행 중 (pid $(pgrep -f full_run.sh | head -1))" \
  || echo "  full_run.sh 없음 (완료 또는 미시작)"
docker ps --format '  {{.Names}}  {{.Status}}' | grep -E "vgrl_t" || true
pgrep -f train_sft.py >/dev/null && echo "  SFT 학습 중"
pgrep -f train_rl.py  >/dev/null && echo "  RL 학습 중"
pgrep -f infer_vllm.py >/dev/null && echo "  추론 중"

echo; echo "=== 현재 단계 ==="
grep -E "^########" "$W/logs/full_run.log" 2>/dev/null | tail -4 | sed 's/^/  /'

echo; echo "=== 산출물 ==="
for f in "$W/data/nusc_cot_$TAG.json" "$W/data/dlm_cot_$TAG.json" "$W/data/cot_$TAG.json" \
         "$W/saves/sft_$TAG/model.safetensors" "$W/saves/rl_viewground_$TAG/model.safetensors" \
         "$W/saves/rl_off_$TAG/model.safetensors"; do
  if [[ -s "$f" ]]; then printf "  OK   %-58s %s\n" "$(basename "$(dirname "$f")")/$(basename "$f")" "$(du -h "$f" | cut -f1)"
  else printf "  --   %s\n" "$(basename "$(dirname "$f")")/$(basename "$f")"; fi
done

echo; echo "=== teacher 생성 진척 ==="
for d in "$W/cot/nusc_$TAG" "$W/cot/dlm_$TAG"; do
  [[ -d $d ]] || continue
  for p in pass1 pass2; do
    [[ -s $d/$p.jsonl ]] && echo "  $(basename $d)/$p: $(wc -l < $d/$p.jsonl) 건"
  done
done

echo; echo "=== 학습 진척 ==="
for lg in "$W/logs/sft_$TAG.log" "$W/logs"/rl_*.log; do
  [[ -s "$lg" ]] || continue
  last=$(tr '\r' '\n' < "$lg" | grep -E "^\s*[0-9]+%\|" | tail -1 | cut -c1-60)
  [[ -n "$last" ]] && echo "  $(basename "$lg"): $last"
done

echo; echo "=== 평가 결과 (나온 것만) ==="
for lg in "$W/logs"/eval_*_uniad.log "$W/logs"/eval_*_stp3.log; do
  [[ -s "$lg" ]] || continue
  row=$(tr '\r' '\n' < "$lg" | grep -vE "^\s*[0-9]+%\||no platform|^\s*$|it/s\]|^Method|^\s+1s" | tail -1)
  [[ -n "$row" ]] && printf "  %-34s %s\n" "$(basename "$lg" .log)" "$row"
done
