"""Blank-view sensitivity: does the model actually read the view that holds the answer?

This is the measurement the method stands or falls on, and it is independent of the
training objective — it never touches kl1/kl2, it just asks the finished model three
questions and compares the answers:

    score_full      all six views
    score_mask_ev   the evidence view blacked out
    score_mask_ctrl one non-evidence view blacked out

    drop_evidence = score_full - score_mask_ev     should be LARGE
    drop_control  = score_full - score_mask_ctrl   should be SMALL
    grounding_gap = drop_evidence - drop_control   <- the headline number

A model answering from language priors has both drops near zero. A model that is
merely brittle to missing pixels has both drops large — that is the PAPO failure
mode, and it shows up here as a large drop_evidence with an equally large
drop_control, i.e. a gap near zero. Only genuine view-specific grounding moves the
gap.

Why this and not nuScenes L2: the RL stage trains on DriveLM QA while L2 scores
nuScenes trajectories, so L2 cannot see what the perception term changes. It also
means the number is comparable across checkpoints that never saw a trajectory.

Exactly one view is blanked in each masked branch, and which one is drawn from a
per-sample seed, so the two branches differ in *which* view was removed and not in
how much was removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vgrl.prompt_format import extract_answer  # noqa: E402
from vgrl.rewards import _f1, object_reference_reward  # noqa: E402
from vgrl.views import CAMERAS  # noqa: E402

BLACK = None  # lazily built, cached per size


def black_like(img: "Image.Image") -> "Image.Image":
    return Image.new("RGB", img.size, (0, 0, 0))


def build_prompt(sample: dict) -> str:
    """Rebuild the chat prompt the RL dataset rows describe."""
    parts = []
    for msg in sample["prompt"]:
        text = " ".join(
            c["text"] for c in msg["content"] if c.get("type") == "text"
        )
        n_img = sum(1 for c in msg["content"] if c.get("type") == "image")
        body = ("<|vision_start|><|image_pad|><|vision_end|>" * n_img) + text
        parts.append(f"<|im_start|>{msg['role']}\n{body}<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n"


def load_views(paths: list[str], resolution: int) -> list["Image.Image"]:
    import math

    out = []
    for p in paths:
        im = Image.open(p)
        if im.width * im.height > resolution:
            f = math.sqrt(resolution / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)), Image.BICUBIC)
        out.append(im.convert("RGB"))
    return out


def pick_views(evidence: list[int], n_views: int, seed: str) -> tuple[int, int] | None:
    """One evidence view and one control view, deterministically per sample."""
    ev = sorted({v for v in evidence if 0 <= v < n_views})
    ctrl = [v for v in range(n_views) if v not in set(ev)]
    if not ev or not ctrl:
        return None
    rng = random.Random(seed)
    return rng.choice(ev), rng.choice(ctrl)


def score(pred: str, gt: str) -> dict:
    ans = extract_answer(pred)
    return {
        "f1": _f1(ans, gt),
        "obj": object_reference_reward([ans], solution=[gt])[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="RL-format DriveLM json with evidence_views")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--image_resolution", type=int, default=401408)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", default=None,
                    help="restrict to one DriveLM category. Mixing them hides the "
                         "signal: measured per-category drop_evidence on the SFT "
                         "model was +0.084 for perception but -0.011 for prediction "
                         "and +0.009 for planning, so the pooled number (~+0.015) "
                         "is prediction and planning diluting perception.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = json.load(open(args.dataset))
    rows = [r for r in rows if r.get("vg_usable", True)]
    if args.category:
        rows = [r for r in rows if r.get("category") == args.category]
        print(f"category={args.category}: {len(rows)} rows")
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

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": len(CAMERAS)},
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    # greedy: the three branches must differ only by the masked view, not by sampling
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=False,
        stop_token_ids=[151645, 151643],
    )

    results = []
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        inputs, meta = [], []
        for r, (ev, ctrl) in chunk:
            views = load_views(r["images"], args.image_resolution)
            prompt = build_prompt(r)
            for branch, blank in (("full", None), ("mask_ev", ev), ("mask_ctrl", ctrl)):
                imgs = list(views)
                if blank is not None:
                    imgs[blank] = black_like(imgs[blank])
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})
                meta.append((r, branch, ev, ctrl))

        outs = llm.generate(inputs, sampling)
        per_id: dict[str, dict] = {}
        for (r, branch, ev, ctrl), o in zip(meta, outs):
            rec = per_id.setdefault(
                r["id"],
                {"id": r["id"], "category": r.get("category"), "evidence_views": r["evidence_views"],
                 "masked_evidence": ev, "masked_control": ctrl},
            )
            rec[branch] = score(o.outputs[0].text, r["solution"])
            # 200자 절단은 공식 Match F1 을 망가뜨렸다 — 좌표가 문장 후반에 나오는데
            # 그 앞에서 잘려 match F1 이 0.3~0.8 (논문 baseline 34.5) 로 붕괴했다.
            rec[f"{branch}_text"] = extract_answer(o.outputs[0].text)[:1200]
        results.extend(per_id.values())
        print(f"  {len(results)}/{len(samples)}", flush=True)

    complete = [r for r in results if all(b in r for b in ("full", "mask_ev", "mask_ctrl"))]
    summary = {"model": args.model, "n": len(complete), "seed": args.seed,
               "batch_size": args.batch_size, "category": args.category}
    for metric in ("f1", "obj"):
        full = sum(r["full"][metric] for r in complete) / max(len(complete), 1)
        mev = sum(r["mask_ev"][metric] for r in complete) / max(len(complete), 1)
        mctrl = sum(r["mask_ctrl"][metric] for r in complete) / max(len(complete), 1)
        summary[metric] = {
            "full": round(full, 4),
            "mask_evidence": round(mev, 4),
            "mask_control": round(mctrl, 4),
            "drop_evidence": round(full - mev, 4),
            "drop_control": round(full - mctrl, 4),
            "grounding_gap": round((full - mev) - (full - mctrl), 4),
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_sample": complete}, open(args.out_json, "w"))

    print("\n" + "=" * 66)
    print(f"model: {args.model}   n={summary['n']}")
    for metric in ("f1", "obj"):
        s = summary[metric]
        print(f"  [{metric}] full={s['full']:.4f}  mask_ev={s['mask_evidence']:.4f}  "
              f"mask_ctrl={s['mask_control']:.4f}")
        print(f"        drop_evidence={s['drop_evidence']:+.4f}  "
              f"drop_control={s['drop_control']:+.4f}  "
              f"GROUNDING GAP={s['grounding_gap']:+.4f}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
