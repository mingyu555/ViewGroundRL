#!/bin/bash
# Serve the CoT teacher on GPUs 4-7 as an OpenAI-compatible endpoint.
#
#   scripts/serve_teacher.sh                      # Qwen2.5-VL-72B, TP=4
#   TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
#     TP=1 GPUS=4 scripts/serve_teacher.sh        # cheap dry-run teacher
#
# 72B bf16 is ~145 GB of weights; four H200s (141 GB each) hold it comfortably at
# TP=4. Check disk before pulling: the weights alone are ~145 GB.
set -euo pipefail

MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
GPUS="${GPUS:-4,5,6,7}"
TP="${TP:-$(tr ',' '\n' <<<"$GPUS" | grep -c .)}"
PORT="${PORT:-8000}"
CACHE="${HF_HOME:-/mnt/ssd1/vgrl/hf}"
MAX_LEN="${MAX_LEN:-32768}"

mkdir -p "$CACHE"
echo "serving $MODEL  TP=$TP  gpus=$GPUS  port=$PORT  cache=$CACHE"

exec docker run --rm --ipc=host --shm-size=32g \
  --gpus "\"device=${GPUS}\"" \
  -p "${PORT}:8000" \
  -v "$CACHE":/hf \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e XDG_CACHE_HOME=/tmp/.cache \
  -e VLLM_CACHE_ROOT=/tmp/.cache/vllm \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  vgrl:train \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --limit-mm-per-prompt '{"image":6}' \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-requests \
    --host 0.0.0.0 --port 8000
