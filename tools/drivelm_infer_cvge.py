"""CVGE 가 붙은 모델로 DriveLM 4카테고리 추론. 출력 형식은 drivelm_infer.py 와 동일.

vLLM 은 훅으로 주입한 모듈을 모르기 때문에 HF generate 를 쓴다. 느리므로 GPU 당
프로세스 하나로 나눠 돌린다 (--shard/--num_shards).

Qwen 은 6뷰 개별 이미지를, VGGT 는 같은 6뷰 원본을 본다 — 학습 때와 같다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_sft_cvge import vggt_preprocess  # noqa: E402
from vgrl.cvge import CVGE, load_vggt, vggt_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvge_dir", required=True)
    ap.add_argument("--model", default=None, help="없으면 cvge_config.json 의 값")
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/dlm_eval_4cat_1200.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_pixels", type=int, default=401408)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--no_cvge", action="store_true", help="CVGE 를 끄고 같은 경로로 추론")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    meta = json.load(open(os.path.join(args.cvge_dir, "cvge_config.json")))
    model_id = args.model or meta["model"]

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, max_pixels=args.max_pixels, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()

    vggt = cvge = None
    if not args.no_cvge:
        cvge = CVGE(hidden_size=meta["hidden_size"], num_layers=meta["num_layers"],
                    in_dim=meta.get("in_dim", 2048), scale=meta.get("scale", 4),
                    num_heads=meta.get("num_heads", 8), dropout=0.0)
        cvge.load_state_dict(torch.load(os.path.join(args.cvge_dir, "cvge.pt"),
                                        map_location="cpu"))
        cvge = cvge.to(dtype=torch.bfloat16).cuda().eval()
        cvge.attach(model.model.language_model.layers)
        vggt = load_vggt(meta["vggt_weights"], device="cuda", dtype=torch.bfloat16)

    image_token_id = model.config.image_token_id

    rows = json.load(open(args.dataset))
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
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
    have = {x["id"] for x in done}
    rows = [r for r in rows if r["id"] not in have]
    print(f"문항 {len(rows)} (이미 완료 {len(have)})", flush=True)

    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        texts, images, views = [], [], []
        for r in chunk:
            texts.append(processor.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True))
            images.append([Image.open(x).convert("RGB") for x in r["images"]])
            views.append(vggt_preprocess(r["images"]))

        enc = processor(text=texts, images=images, padding=True,
                        return_tensors="pt").to("cuda")
        if cvge is not None:
            f3d = vggt_features(vggt, torch.stack(views).to("cuda", torch.bfloat16))
            cvge.set_batch(f3d, enc["input_ids"] == image_token_id)
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
        finally:
            if cvge is not None:
                cvge.clear_batch()

        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(chunk, processor.batch_decode(gen, skip_special_tokens=True)):
            done.append({"id": r["id"], "category": r["category"],
                         "question": r.get("question", ""),
                         "solution": r["solution"],
                         "llm-response": g.strip()})
        print(f"  {len(done)}/{len(rows)}", flush=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
    json.dump(done, open(args.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
