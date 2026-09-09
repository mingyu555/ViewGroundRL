#!/usr/bin/env bash
# DriveLMM-o1 평가를 GPU 4장에 데이터 병렬로 뿌린다.
# 8B 모델은 H200 한 장에 다 올라가므로, 텐서 병렬(느림)이 아니라 GPU 당 전체 사본을
# 띄우고 데이터를 1/4 씩 나눠 맡긴다.
#   $1 스크립트  $2 모델경로  $3 출력접두사  $4 태그  $5 GPU목록(예 0,1,2,3)
#   (나머지 인자는 하위 스크립트로 그대로 전달)
set -e
SCRIPT=$1; MODEL=$2; OUTPREFIX=$3; TAG=$4; GPUS=$5; shift 5
IFS=, read -ra GPUARR <<< "$GPUS"
PIDS=()
for i in "${!GPUARR[@]}"; do
  g=${GPUARR[$i]}
  docker rm -f "vgrl_${TAG}_$i" >/dev/null 2>&1 || true
  CONTAINER_NAME="vgrl_${TAG}_$i" ./scripts/dr.sh "$g" python3 "$SCRIPT" \
    --model "$MODEL" \
    --dataset /mnt/ssd4/mingyu/vgrl/data/dlmm_test.json \
    --cache_dir /mnt/ssd4/mingyu/dlmm_stitch \
    --out "${OUTPREFIX}_s$i.json" \
    --shard "$i" --num_shards "${#GPUARR[@]}" \
    "$@" > "/mnt/ssd4/mingyu/vgrl/logs/${TAG}_s$i.log" 2>&1 &
  PIDS+=($!)
done
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=1; done
python3 - "$OUTPREFIX" <<'PY'
import json, sys, glob, os
pre = sys.argv[1]
merged, seen = [], set()
for f in sorted(glob.glob(pre + "_s*.json")):
    for r in json.load(open(f)):
        if r["idx"] in seen: continue
        seen.add(r["idx"]); merged.append(r)
out = pre + ".json"
json.dump(merged, open(out, "w"), ensure_ascii=False)
print(f"합침: {len(merged)}건 -> {out}")
PY
exit $fail
