"""DriveLM 4개 카테고리 추론. 6뷰를 따로 넣는다 (DriveLMM-o1 의 격자 방식과 다름).

DriveLM 정답은 '<c1,CAM_FRONT,450.5,355.5>' 처럼 카메라 이름을 직접 쓰므로 6뷰를
따로 주는 게 맞다. 격자로 합치면 카메라 이름과 위치 대응이 깨진다.

Qwen2.5-VL 과 Qwen3-VL 은 채팅 템플릿이 동일해서(확인함) 같은 코드로 공정 비교가
된다. 프롬프트는 프로세서의 apply_chat_template 로 만들어 모델별 차이를 흡수한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os


def pick_control(evidence, n_views, seed_key):
    """근거뷰가 아닌 뷰 하나를 표본마다 결정론적으로 고른다."""
    import random
    ev = {v for v in evidence if 0 <= v < n_views}
    ctrl = [v for v in range(n_views) if v not in ev]
    if not ev or not ctrl:
        return None
    return random.Random(seed_key).choice(ctrl)


def blackout(img):
    from PIL import Image
    return Image.new("RGB", img.size, (0, 0, 0))


def load_views(paths, max_px):
    from PIL import Image
    out = []
    for p in paths:
        im = Image.open(p)
        if im.width * im.height > max_px:
            f = math.sqrt(max_px / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)), Image.BICUBIC)
        out.append(im.convert("RGB"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/dlm_eval_4cat_1200.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--image_resolution", type=int, default=401408, help="뷰 하나당 상한 픽셀")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.42)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default=None)
    ap.add_argument("--mask", choices=["none", "evidence", "control"], default="none",
                    help="6뷰 중 하나를 검게 칠한다 (blank-view 진단). evidence 는 "
                         "정답이 의존하는 뷰, control 은 그 외 뷰 하나를 고른다.")
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

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
    if not rows:
        return 0

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": 6}, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              mm_processor_kwargs={"max_pixels": args.image_resolution},
              disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=True)

    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        inputs, keep = [], []
        for r in chunk:
            views = load_views(r["images"], args.image_resolution)
            if args.mask != "none":
                ev = r.get("evidence_views") or []
                if args.mask == "evidence":
                    cell = ev[0] if ev else None
                else:
                    cell = pick_control(ev, len(views), f"{r['id']}:{args.mask_seed}")
                if cell is None:
                    continue                  # 근거/대조 분리가 안 되는 행은 건너뛴다
                views = list(views)
                views[cell] = blackout(views[cell])
            text = proc.apply_chat_template(r["prompt"], tokenize=False,
                                            add_generation_prompt=True)
            inputs.append({"prompt": text,
                           "multi_modal_data": {"image": views}})
            keep.append(r)
        if not inputs:
            continue
        for r, o in zip(keep, llm.generate(inputs, sampling)):
            done.append({"id": r["id"], "category": r["category"],
                         "question": r.get("question", ""),
                         "solution": r["solution"],
                         "llm-response": o.outputs[0].text.strip()})
        print(f"  {len(done)}/{len(rows)}", flush=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
    json.dump(done, open(args.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
