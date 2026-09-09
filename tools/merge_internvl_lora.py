"""ayeshaishaq/DriveLMMo1 의 미병합 LoRA 체크포인트를 병합한다.

공개된 체크포인트는 PEFT 로 감싼 상태 그대로 저장돼 있어서 vLLM 이 못 읽는다:
    language_model.base_model.model.model.layers.0.attention.wo.base_layer.weight
    language_model.base_model.model.model.layers.0.attention.wo.lora_A.default.weight
    language_model.base_model.model.model.layers.0.attention.wo.lora_B.default.weight

modeling_internvl_chat.py:96 이 wrap_llm_lora(r=use_llm_lora, lora_alpha=2*use_llm_lora)
로 감싸고 config 의 use_llm_lora=16 이므로 r=16, alpha=32, 스케일 2.0 이다.
use_backbone_lora=0 이라 비전 쪽은 LoRA 가 없다.

    W_merged = W_base + (alpha / r) * (B @ A)

키는 PEFT 래퍼 경로를 벗겨 원래 이름으로 되돌린다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "language_model.base_model.model."
CLEAN_PREFIX = "language_model."


def clean_key(k: str) -> str:
    return CLEAN_PREFIX + k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard_size_gb", type=float, default=4.5)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    r = cfg["use_llm_lora"]
    alpha = 2 * r                      # modeling_internvl_chat.py:96
    scale = alpha / r
    assert cfg.get("use_backbone_lora", 0) == 0, "비전 백본 LoRA 는 처리 대상이 아니다"
    print(f"LoRA r={r} alpha={alpha} scale={scale}")

    sd = {}
    for f in sorted(glob.glob(os.path.join(args.src, "*.safetensors"))):
        sd.update(load_file(f))
    print(f"원본 텐서 {len(sd)}")

    out, merged, passthrough = {}, 0, 0
    for k, v in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue                                    # 아래에서 base_layer 와 함께 처리
        if k.endswith(".base_layer.weight"):
            stem = k[: -len(".base_layer.weight")]
            a = sd[f"{stem}.lora_A.default.weight"]
            b = sd[f"{stem}.lora_B.default.weight"]
            assert a.shape[0] == r and b.shape[1] == r, f"랭크 불일치 {k}: {a.shape} {b.shape}"
            delta = (b.float() @ a.float()) * scale
            assert delta.shape == v.shape, f"형상 불일치 {k}: {delta.shape} vs {v.shape}"
            out[clean_key(f"{stem}.weight")] = (v.float() + delta).to(v.dtype)
            merged += 1
        else:
            out[clean_key(k)] = v
            passthrough += 1
    print(f"병합 {merged}개, 그대로 {passthrough}개 -> 총 {len(out)}")

    os.makedirs(args.dst, exist_ok=True)
    limit = int(args.shard_size_gb * 1e9)
    shards, cur, cur_sz = [], {}, 0
    for k, v in out.items():
        nb = v.numel() * v.element_size()
        if cur and cur_sz + nb > limit:
            shards.append(cur); cur, cur_sz = {}, 0
        cur[k] = v; cur_sz += nb
    if cur:
        shards.append(cur)

    weight_map, total = {}, 0
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(args.dst, name), metadata={"format": "pt"})
        for k, v in sh.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
        print(f"  {name}: 텐서 {len(sh)}")
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=2)

    for f in os.listdir(args.src):
        if f.endswith((".safetensors", ".orig")) or f == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, f))

    # 병합본에는 PEFT 래퍼가 없으니 config 의 LoRA 플래그를 꺼야 로딩 때 다시 감싸지 않는다
    c = json.load(open(os.path.join(args.dst, "config.json")))
    c["use_llm_lora"] = 0
    json.dump(c, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
    print(f"완료 -> {args.dst}  (use_llm_lora 를 0 으로 내림)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
