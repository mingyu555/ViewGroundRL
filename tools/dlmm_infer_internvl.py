"""DriveLMM-o1 공식 모델(InternVL2.5-8B 기반) 추론.

ayeshaishaq/DriveLMMo1 는 Qwen 계열이 아니라 InternVLChatModel 이라 프롬프트
구성이 완전히 다르다. 논문 표의 62.36 을 재현하려면 공식 evaluation/inference.py
와 같은 형식이어야 해서, 거기서 그대로 옮겼다:

  prompt = "When answering the question based on the provided image, ..."
  question = prompt + "\\n<image>\\n" + question           (inference.py:209)
  generation: max_new_tokens 2000, do_sample False         (inference.py:271)

대화 템플릿은 모델 config 의 template=internvl2_5 를 따른다. 토크나이저의
chat_template 에는 시스템 메시지가 없어서 conversation.py 의 정의를 직접 쓴다
(모델의 .chat() 이 쓰는 것과 같은 문자열).

이미지는 우리 다른 평가와 같은 2x3 격자 한 장. 동적 타일링은 vLLM 이 config 의
max_dynamic_patch=6 / use_thumbnail=True 를 읽어 알아서 한다 — 공식 max_num=6 과 같다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dlmm_infer import mask_cell, pick_control, stitch  # noqa: E402

# inference.py:244 축자 이식
OFFICIAL_PROMPT = (
    "When answering the question based on the provided image, follow a structured "
    "and logical reasoning process. Organize your response using the format, "
    "ensuring each step builds upon the previous one and clearly explains how the "
    "image(s) contribute to the solution. Your answer should be structured as "
    "Reasoning Steps: (step by step reasoning) Final Answer: (final answer) \n Question: "
)

# conversation.py:384-390, template name internvl2_5
SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作"
    "单位联合开发的多模态大语言模型。"
)


def build_prompt(question: str) -> str:
    q = OFFICIAL_PROMPT + "\n<image>\n" + question
    return (f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/dlmm_test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="/mnt/ssd4/mingyu/dlmm_stitch")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2000)   # 공식값
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.42)
    ap.add_argument("--limit", type=int, default=0)
    # 데이터 병렬: GPU 당 프로세스 하나가 전체의 1/num_shards 를 맡는다.
    # 8B 모델은 한 장에 다 올라가므로 텐서 병렬보다 처리량이 훨씬 낫다.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--mask", choices=["none", "evidence", "control"], default="none")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    rows = json.load(open(args.dataset))
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard :: args.num_shards]
    done = []
    if os.path.exists(args.out):
        try:
            done = json.load(open(args.out))
        except Exception:
            done = []
    have = {x["idx"] for x in done}
    rows = [r for r in rows if r["id"].split("::", 1)[1] not in have]
    print(f"문항 {len(rows)} (이미 완료 {len(have)})", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": 1}, max_model_len=8192,
              gpu_memory_utilization=args.gpu_memory_utilization,
              disable_log_stats=True)
    # 공식은 temperature 0 일 때 do_sample=False. 그리디로 맞춘다.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=True)

    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        inputs, keep = [], []
        for r in chunk:
            key = r["id"].split("::", 1)[1].rsplit("_", 1)[0]
            img = Image.open(stitch(r["views"], args.cache_dir, key)).convert("RGB")
            if args.mask != "none":
                ev = r.get("evidence_views") or []
                cell = ev[0] if args.mask == "evidence" else pick_control(ev, r["id"])
                if cell is None:
                    continue
                img = mask_cell(img, cell)
            inputs.append({"prompt": build_prompt(r["prompt_text"]),
                           "multi_modal_data": {"image": img}})
            keep.append(r)
        if not inputs:
            continue
        for r, o in zip(keep, llm.generate(inputs, sampling)):
            done.append({"idx": r["id"].split("::", 1)[1],
                         "question": r["prompt_text"],
                         "llm-response": o.outputs[0].text.strip()})
        print(f"  {len(done)}/{len(rows)}", flush=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
    json.dump(done, open(args.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
