#!/usr/bin/env bash
# 평가를 GPU 여러 장에 데이터 병렬로 뿌린다 (GPU 당 모델 사본 하나).
#   $1 스크립트  $2 모델경로  $3 출력접두사  $4 태그  $5 GPU목록(예 0,1,2,3,4,5,6,7)
#   나머지 인자는 하위 스크립트로 그대로 전달
set -e
SCRIPT=$1; MODEL=$2; OUTPREFIX=$3; TAG=$4; GPUS=$5; shift 5
IFS=, read -ra G <<< "$GPUS"
PIDS=()
for i in "${!G[@]}"; do
  docker rm -f "vgrl_${TAG}_$i" >/dev/null 2>&1 || true
  CONTAINER_NAME="vgrl_${TAG}_$i" ./scripts/dr.sh "${G[$i]}" python3 "$SCRIPT" \
    --model "$MODEL" \
    --out "${OUTPREFIX}_s$i.json" \
    --shard "$i" --num_shards "${#G[@]}" \
    "$@" > "/mnt/ssd4/mingyu/vgrl/logs/${TAG}_s$i.log" 2>&1 &
  PIDS+=($!)
done
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=1; done
python3 - "$OUTPREFIX" <<'PY'
import json, sys, glob
pre = sys.argv[1]
merged, seen = [], set()
for f in sorted(glob.glob(pre + "_s*.json")):
    for r in json.load(open(f)):
        if r["id"] in seen: continue
        seen.add(r["id"]); merged.append(r)
json.dump(merged, open(pre + ".json", "w"), ensure_ascii=False)
print(f"합침: {len(merged)}건 -> {pre}.json")
PY
exit $fail
