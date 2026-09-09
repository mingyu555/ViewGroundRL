"""What does the untrained Qwen2.5-VL actually respond to?

The base model scores L2 9.86 m on nuScenes planning with a 7.75% parse rate, so
its blank-view numbers were uninterpretable — you cannot measure whether a model
reads the right camera when it cannot do the task at all. This asks the prior
question: which parts of the input move its behaviour, and by how much.

Each variant strips or blanks one input channel and reports two things:

  parse_rate  fraction of generations yielding six waypoints. For this model this
              is the dominant axis — format compliance, not planning.
  L2          masked L2 over the parseable subset. Read with care: the subset
              differs between variants, so a variant that parses rarely is scored
              on an easier, self-selected slice.

    python tools/base_ablation.py --model <base> --dataset <rl-format val json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.prompt_format import extract_answer  # noqa: E402
from vgrl.views import CAMERAS  # noqa: E402

WAYPOINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
HIST_RE = re.compile(r"Historical Trajectory \(last 2 seconds\): \[[^\]]*\]\n?")
GOAL_RE = re.compile(r"Mission Goal: \w+\n?")
SPEED_RE = re.compile(r"Current longitudinal speed: [^\n]*\n?")
ACCEL_RE = re.compile(r"Current longitudinal acceleration: [^\n]*\n?")
RULES_RE = re.compile(r"Traffic Rules:.*?occupied regions\.\n?", re.S)

# Each variant: (label, text transform, blank_images)
VARIANTS = [
    ("A. 전체 (기준)",            lambda t: t,                               False),
    ("B. 이미지 전부 검게",        lambda t: t,                               True),
    ("C. 과거 궤적 제거",          lambda t: HIST_RE.sub("", t),              False),
    ("D. 속도/가속도 제거",        lambda t: ACCEL_RE.sub("", SPEED_RE.sub("", t)), False),
    ("E. 미션 목표 제거",          lambda t: GOAL_RE.sub("", t),              False),
    ("F. 교통 규칙 제거",          lambda t: RULES_RE.sub("", t),             False),
    ("G. 텍스트 문맥 전부 제거",    lambda t: RULES_RE.sub("", GOAL_RE.sub("", ACCEL_RE.sub(
                                      "", SPEED_RE.sub("", HIST_RE.sub("", t))))), False),
    ("H. 이미지만 (텍스트 문맥 제거)", lambda t: RULES_RE.sub("", GOAL_RE.sub("", ACCEL_RE.sub(
                                      "", SPEED_RE.sub("", HIST_RE.sub("", t))))), False),
    ("I. 아무 단서 없음",          lambda t: RULES_RE.sub("", GOAL_RE.sub("", ACCEL_RE.sub(
                                      "", SPEED_RE.sub("", HIST_RE.sub("", t))))), True),
]


def build_prompt(sample: dict, transform, n_img: int) -> str:
    parts = []
    for msg in sample["prompt"]:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        text = transform(text)
        imgs = sum(1 for c in content if c.get("type") == "image")
        body = ("<|vision_start|><|image_pad|><|vision_end|>" * imgs) + text
        parts.append(f"<|im_start|>{msg['role']}\n{body}<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n"


def load_views(paths, resolution):
    out = []
    for p in paths:
        im = Image.open(p)
        if im.width * im.height > resolution:
            f = math.sqrt(resolution / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)), Image.BICUBIC)
        out.append(im.convert("RGB"))
    return out


def masked_l2(text, gt, mask):
    pairs = WAYPOINT_RE.findall(extract_answer(text))
    if len(pairs) < len(gt):
        return None
    p = np.array([(float(a), float(b)) for a, b in pairs[: len(gt)]], dtype=float)
    keep = mask[:, 0] > 0
    if not keep.any():
        return None
    return float(np.linalg.norm(p[keep] - gt[keep], axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gt_folder", default="/mnt/ssd1/vgrl/metrics")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--image_resolution", type=int, default=401408)
    ap.add_argument("--max_new_tokens", type=int, default=448)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    gt_all = pickle.load(open(os.path.join(args.gt_folder, "gt_traj.pkl"), "rb"))
    mask_all = pickle.load(open(os.path.join(args.gt_folder, "gt_traj_mask.pkl"), "rb"))

    rows = [r for r in json.load(open(args.dataset)) if r["id"] in gt_all]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_samples]
    print(f"{len(rows)} frames x {len(VARIANTS)} variants", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=1, limit_mm_per_prompt={"image": len(CAMERAS)},
              max_model_len=8192, gpu_memory_utilization=args.gpu_memory_utilization,
              disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=False, stop_token_ids=[151645, 151643])

    per_variant = {label: {"l2": [], "parsed": 0, "n": 0} for label, _, _ in VARIANTS}

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        inputs, meta = [], []
        for r in chunk:
            views = load_views(r["images"], args.image_resolution)
            black = [Image.new("RGB", im.size, (0, 0, 0)) for im in views]
            for label, transform, blank in VARIANTS:
                imgs = black if blank else views
                inputs.append({"prompt": build_prompt(r, transform, len(imgs)),
                               "multi_modal_data": {"image": list(imgs)}})
                meta.append((r, label))

        outs = llm.generate(inputs, sampling)
        for (r, label), o in zip(meta, outs):
            gt = np.asarray(gt_all[r["id"]]).reshape(-1, 2)
            mk = np.asarray(mask_all[r["id"]]).reshape(-1, 2)
            v = masked_l2(o.outputs[0].text, gt, mk)
            per_variant[label]["n"] += 1
            if v is not None:
                per_variant[label]["parsed"] += 1
                per_variant[label]["l2"].append(v)
        print(f"  {min(start+args.batch_size, len(rows))}/{len(rows)} frames", flush=True)

    summary = {}
    for label, _, _ in VARIANTS:
        d = per_variant[label]
        summary[label] = {
            "n": d["n"], "parse_rate": round(d["parsed"] / max(d["n"], 1), 4),
            "l2_on_parsed": round(sum(d["l2"]) / len(d["l2"]), 4) if d["l2"] else None,
            "n_parsed": d["parsed"],
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"model": args.model, "summary": summary}, open(args.out_json, "w"),
              ensure_ascii=False, indent=1)

    print("\n" + "=" * 74)
    print(f"{'변형':<26}{'파싱률':>9}{'파싱건수':>10}{'L2(파싱분)':>13}")
    print("-" * 74)
    base = summary["A. 전체 (기준)"]
    for label, _, _ in VARIANTS:
        s = summary[label]
        l2 = f"{s['l2_on_parsed']:.3f}" if s["l2_on_parsed"] is not None else "  n/a"
        dp = s["parse_rate"] - base["parse_rate"]
        print(f"{label:<26}{s['parse_rate']*100:>8.1f}%{s['n_parsed']:>10}{l2:>13}"
              f"   (파싱률 {dp*100:+.1f}pt)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
