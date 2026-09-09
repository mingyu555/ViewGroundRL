"""Blank-view sensitivity for nuScenes trajectory planning.

The trajectory counterpart of eval_grounding.py, which scores DriveLM text
answers. Same three-branch design, different metric: L2 to the ground-truth
waypoints instead of token F1.

    L2_full       all six views
    L2_mask_ev    the evidence view blacked out
    L2_mask_ctrl  one non-evidence view blacked out

    drop_evidence = L2_mask_ev   - L2_full     should be LARGE (worse without it)
    drop_control  = L2_mask_ctrl - L2_full     should be SMALL
    grounding_gap = drop_evidence - drop_control        <- the headline number

Note the sign convention differs from the DriveLM script: there the metric was
accuracy (higher better) so a "drop" was a decrease; here it is L2 (lower better)
so a "drop" is an increase. Both are defined so that a well-grounded model has a
large positive drop_evidence.

L2 is masked by `gt_traj_mask`: nuScenes pads unobserved future steps with (0,0),
and scoring them would compare the model against padding. This is the same defect
the training reward has (vgrl.rewards.trajectory_l2_reward ignores the mask) —
fixed here so the measurement at least is clean.

Exactly one view is blanked per masked branch, drawn from a per-sample seed, so
the branches differ in *which* view was removed and not in how much.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vgrl.prompt_format import extract_answer  # noqa: E402
from vgrl.views import CAMERAS  # noqa: E402

WAYPOINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")

# Option A gate: strip the ego channels from the prompt. Measured input ablations
# put past-trajectory at +2.76 m and speed/accel at +1.05 m of L2 against +0.11 m
# for all six images, so with ego state present the perception term is competing
# for ~4% of the signal. VAD-Base reports the same all-blank ablation going from
# +0.09 m (with ego) to +3.08 m (without) — arXiv:2312.03031.
HIST_RE = re.compile(r"Historical Trajectory \(last 2 seconds\): \[[^\]]*\]\n?")
SPEED_RE = re.compile(r"Current longitudinal speed: [^\n]*\n?")
ACCEL_RE = re.compile(r"Current longitudinal acceleration: [^\n]*\n?")


def strip_ego(text: str) -> str:
    return ACCEL_RE.sub("", SPEED_RE.sub("", HIST_RE.sub("", text)))


def build_prompt(sample: dict, no_ego: bool = False) -> str:
    parts = []
    for msg in sample["prompt"]:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if no_ego:
            text = strip_ego(text)
        n_img = sum(1 for c in content if c.get("type") == "image")
        body = ("<|vision_start|><|image_pad|><|vision_end|>" * n_img) + text
        parts.append(f"<|im_start|>{msg['role']}\n{body}<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n"


def load_views(paths: list[str], resolution: int) -> list["Image.Image"]:
    out = []
    for p in paths:
        im = Image.open(p)
        if im.width * im.height > resolution:
            f = math.sqrt(resolution / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)), Image.BICUBIC)
        out.append(im.convert("RGB"))
    return out


def pick_views(evidence: list[int], n_views: int, seed: str):
    ev = sorted({v for v in evidence if 0 <= v < n_views})
    ctrl = [v for v in range(n_views) if v not in set(ev)]
    if not ev or not ctrl:
        return None
    rng = random.Random(seed)
    return rng.choice(ev), rng.choice(ctrl)


def l2(pred_text: str, gt: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean masked L2, or None when the prediction is unusable."""
    pairs = WAYPOINT_RE.findall(extract_answer(pred_text))
    if len(pairs) < len(gt):
        return None
    p = np.array([(float(a), float(b)) for a, b in pairs[: len(gt)]], dtype=float)
    keep = mask[:, 0] > 0
    if not keep.any():
        return None
    d = np.linalg.norm(p[keep] - gt[keep], axis=1)
    return float(d.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="RL-format nuScenes val json")
    ap.add_argument("--gt_folder", default="/mnt/ssd1/vgrl/metrics")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_samples", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_resolution", type=int, default=401408)
    ap.add_argument("--max_new_tokens", type=int, default=448)
    # 0.88 은 139 GB H200 에서 KV 캐시로 114 GB(3.3M 토큰)를 잡아 활성값 자리를 남기지
    # 않는다. 분기를 4개로 고정한 뒤 동시 멀티모달 입력이 늘어나 OOM 이 났다.
    # 이 과제는 프롬프트가 8k 토큰 이하라 KV 캐시가 그만큼 필요하지 않다.
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.62)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ego", action="store_true",
                    help="strip past trajectory and speed/accel from the prompt")
    ap.add_argument("--blank_all", action="store_true",
                    help="(kept for compatibility; the all-blank branch is now always "
                         "computed so that batch composition — and therefore the bf16 "
                         "greedy path — does not depend on this flag)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    gt_all = pickle.load(open(os.path.join(args.gt_folder, "gt_traj.pkl"), "rb"))
    mask_all = pickle.load(open(os.path.join(args.gt_folder, "gt_traj_mask.pkl"), "rb"))

    rows = [r for r in json.load(open(args.dataset)) if r.get("vg_usable")]
    rows = [r for r in rows if r["id"] in gt_all]
    random.Random(args.seed).shuffle(rows)

    samples = []
    for r in rows:
        pick = pick_views(r["evidence_views"], len(CAMERAS), f"{args.seed}:{r['id']}")
        if pick is None:
            continue
        samples.append((r, pick))
        if len(samples) >= args.max_samples:
            break
    print(f"{len(samples)} samples with a matched evidence/control split", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=1, limit_mm_per_prompt={"image": len(CAMERAS)},
              max_model_len=8192, gpu_memory_utilization=args.gpu_memory_utilization,
              disable_log_stats=True)
    # greedy: the branches must differ only by the masked view, not by sampling
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=False, stop_token_ids=[151645, 151643])

    results = []
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        inputs, meta = [], []
        for r, (ev, ctrl) in chunk:
            views = load_views(r["images"], args.image_resolution)
            prompt = build_prompt(r, no_ego=args.no_ego)
            # 분기 수를 항상 4로 고정한다. 이전 버전은 --blank_all 여부에 따라 3개
            # 또는 4개를 넣었고, 배치 구성이 달라지면 bf16 누적 순서가 바뀌어 경계선상
            # 토큰의 greedy 경로가 갈렸다. 같은 모델·같은 시드로 두 번 측정했을 때
            # drop_evidence 가 sft +0.0365 vs +0.0240, papo 는 부호까지 뒤집혔다.
            # 실행 간 변동이 모델 간 차이와 같은 크기여서 어떤 순위도 신뢰할 수 없었다.
            branches = [("full", None), ("mask_ev", ev), ("mask_ctrl", ctrl),
                        ("mask_all", "all")]
            for branch, blank in branches:
                imgs = list(views)
                if blank == "all":
                    imgs = [Image.new("RGB", im.size, (0, 0, 0)) for im in imgs]
                elif blank is not None:
                    imgs[blank] = Image.new("RGB", imgs[blank].size, (0, 0, 0))
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})
                meta.append((r, branch, ev, ctrl))

        outs = llm.generate(inputs, sampling)
        per_id: dict[str, dict] = {}
        for (r, branch, ev, ctrl), o in zip(meta, outs):
            gt = np.asarray(gt_all[r["id"]]).reshape(-1, 2)
            mk = np.asarray(mask_all[r["id"]]).reshape(-1, 2)
            rec = per_id.setdefault(r["id"], {
                "id": r["id"], "evidence_views": r["evidence_views"],
                "masked_evidence": ev, "masked_control": ctrl,
                "n_observed": int((mk[:, 0] > 0).sum()),
            })
            rec[branch] = l2(o.outputs[0].text, gt, mk)
        results.extend(per_id.values())
        print(f"  {len(results)}/{len(samples)}", flush=True)

    # A sample only counts if all three branches produced a parseable trajectory —
    # otherwise the comparison is between different subsets.
    # 네 분기 전부 파싱된 샘플만 센다. 분기 수를 고정했으므로 --blank_all 여부와
    # 무관하게 같은 부분집합이 되고, 두 실행이 비교 가능해진다.
    needed = ["full", "mask_ev", "mask_ctrl", "mask_all"]
    complete = [r for r in results if all(r.get(b) is not None for b in needed)]
    n_drop = len(results) - len(complete)

    def mean(key):
        return sum(r[key] for r in complete) / max(len(complete), 1)

    full, mev, mctrl = mean("full"), mean("mask_ev"), mean("mask_ctrl")
    summary = {
        "model": args.model, "n": len(complete), "n_unparseable_dropped": n_drop,
        "seed": args.seed, "no_ego": bool(args.no_ego), "batch_size": args.batch_size,
        "l2": {
            "full": round(full, 4),
            "mask_evidence": round(mev, 4),
            "mask_control": round(mctrl, 4),
            # L2 is lower-is-better, so a "drop" in quality is an increase in L2
            "drop_evidence": round(mev - full, 4),
            "drop_control": round(mctrl - full, 4),
            "grounding_gap": round((mev - full) - (mctrl - full), 4),
        },
    }
    if True:
        mall = mean("mask_all")
        summary["l2"]["mask_all_views"] = round(mall, 4)
        summary["l2"]["drop_all_views"] = round(mall - full, 4)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_sample": complete}, open(args.out_json, "w"))

    s = summary["l2"]
    print("\n" + "=" * 68)
    print(f"model: {args.model}   n={summary['n']}  (dropped {n_drop} unparseable)")
    print(f"  L2  full={s['full']:.4f}  mask_ev={s['mask_evidence']:.4f}  "
          f"mask_ctrl={s['mask_control']:.4f}")
    print(f"      drop_evidence={s['drop_evidence']:+.4f}  "
          f"drop_control={s['drop_control']:+.4f}  "
          f"GROUNDING GAP={s['grounding_gap']:+.4f}")
    if "drop_all_views" in s:
        print(f"      ALL SIX BLANK: L2={s['mask_all_views']:.4f}  "
              f"drop={s['drop_all_views']:+.4f}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
