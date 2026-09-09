#!/bin/bash
# Stage 1 — plain SFT on DriveLM + nuScenes. No view-grounding term by design.
#
# The LLaMA-Factory copy inherited from MindDriver is gutted (58 of 64 source files
# are 0 bytes), so install the real one first:
#   pip install "llamafactory[metrics,deepspeed]==0.9.5"
# then register the two datasets from configs/dataset_info_add.json into
# LLaMA-Factory's data/dataset_info.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export FORCE_TORCHRUN=1

llamafactory-cli train configs/sft_stage.yaml
