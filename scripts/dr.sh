#!/bin/bash
# Run a command inside the training image.
#   usage: scripts/dr.sh <gpu-list|none> <cmd...>
# GPUs are host indices. This box is shared — stay on 4,5,6,7.
set -euo pipefail
GPUS="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_ARGS=()
[[ "$GPUS" != "none" ]] && GPU_ARGS=(--gpus "\"device=${GPUS}\"")

# CONTAINER_NAME 을 주면 이름을 붙인다. 이름이 있으면 `docker rm -f <name>` 으로
# 확실히 정리할 수 있다 — 이름 없이 띄운 스모크가 GPU 를 붙잡은 채 남아, 본 실행과
# 겹쳐 OOM 을 낸 적이 있다.
NAME_ARGS=()
[[ -n "${CONTAINER_NAME:-}" ]] && NAME_ARGS=(--name "$CONTAINER_NAME")

exec docker run --rm --ipc=host --shm-size=32g \
  ${NAME_ARGS[@]+"${NAME_ARGS[@]}"} \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  -v "$REPO":/work \
  -v /mnt/ssd1/vgrl:/mnt/ssd1/vgrl \
  -v /mnt/ssd1/omnidrive:/mnt/ssd1/omnidrive:ro \
  -v /mnt/ssd2/mingyu:/mnt/ssd2/mingyu \
  -v /mnt/ssd4/mingyu:/mnt/ssd4/mingyu \
  -v /data/nuScenes:/nuscenes:ro \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  ${NCCL_TIMEOUT_MS:+-e NCCL_TIMEOUT_MS=$NCCL_TIMEOUT_MS} \
  ${TORCH_NCCL_ASYNC_ERROR_HANDLING:+-e TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING} \
  ${TORCH_NCCL_BLOCKING_WAIT:+-e TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT} \
  ${PYTORCH_CUDA_ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF} \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/inductor \
  -e TRITON_CACHE_DIR=/tmp/.cache/triton \
  -e VLLM_CACHE_ROOT=/tmp/.cache/vllm \
  -e HF_HOME=${HF_HOME_OVERRIDE:-/mnt/ssd4/mingyu/hf} \
  -w /work \
  vgrl:train "$@"
