#!/bin/bash
# Serve the CoT teacher as N independent TP=1 servers, one per GPU.
#
# Why not tensor parallel: vllm 0.25.1 with an AWQ checkpoint dies in the
# all-reduce path on this box ("custom_all_reduce.cuh:455 illegal memory access",
# and still an illegal access during CUDA-graph capture with
# --disable-custom-all-reduce). A 43 GB AWQ 72B fits on one 141 GB H200 anyway, so
# data parallelism replaces tensor parallelism — which for bulk generation is also
# faster, since per-token all-reduce disappears entirely.
#
#   scripts/serve_teacher_dp.sh            # GPUs 4,5,6,7 -> ports 8004..8007
#   scripts/serve_teacher_dp.sh stop
#
# The matching client setting is a comma-separated TEACHER_BASE_URL; cot_teacher
# round-robins requests across the list.
set -uo pipefail

MODEL="${TEACHER_MODEL_PATH:-/mnt/ssd1/vgrl/teacher_72b_awq}"
GPUS="${GPUS:-4 5 6 7}"
NAME_PREFIX=vgrl_t
EAGER="${EAGER:-0}"        # EAGER=1 skips CUDA-graph capture

if [[ "${1:-}" == "stop" ]]; then
  for g in $GPUS; do docker rm -f "${NAME_PREFIX}${g}" 2>/dev/null; done
  echo "stopped"
  exit 0
fi

for g in $GPUS; do
  port=$((8000 + g))
  docker rm -f "${NAME_PREFIX}${g}" 2>/dev/null
  extra=()
  [[ "$EAGER" == "1" ]] && extra+=(--enforce-eager)
  docker run -d --rm --name "${NAME_PREFIX}${g}" --ipc=host --shm-size=16g \
    --gpus "\"device=${g}\"" -p "${port}:8000" \
    -v /mnt/ssd1/vgrl:/mnt/ssd1/vgrl \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e VLLM_CACHE_ROOT=/tmp/.cache/vllm \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/inductor -e TRITON_CACHE_DIR=/tmp/.cache/triton \
    vgrl:train python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name teacher-72b \
      --tensor-parallel-size 1 --max-model-len 16384 \
      --limit-mm-per-prompt '{"image":6}' \
      --gpu-memory-utilization 0.88 --trust-remote-code \
      ${extra[@]+"${extra[@]}"} \
      --host 0.0.0.0 --port 8000 > /dev/null
  echo "launched ${NAME_PREFIX}${g} on GPU $g -> :$port"
done

echo "waiting for readiness..."
for g in $GPUS; do
  port=$((8000 + g))
  for _ in $(seq 1 80); do
    curl -s --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q "^${NAME_PREFIX}${g}$" || { echo "  ${NAME_PREFIX}${g} DIED"; break; }
    sleep 15
  done
  curl -s --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 \
    && echo "  ready :$port" || echo "  NOT ready :$port"
done

urls=""
for g in $GPUS; do urls="${urls:+$urls,}http://127.0.0.1:$((8000 + g))/v1"; done
echo
echo "export TEACHER_BASE_URL='$urls'"
echo "export TEACHER_MODEL=teacher-72b"
