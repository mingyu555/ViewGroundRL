"""Build the nuScenes planning RL set, with derived evidence views.

Why this exists: the RL stage so far trained only on DriveLM QA, whose "planning"
category is object-level classification ("is <c4,CAM_FRONT,...> an object the ego
should consider?" -> "Yes."), not trajectory generation. The nuScenes L2/collision
benchmark measures trajectory generation, so the RL stage never touched the task
the benchmark scores — which is why sft / rl_off / rl_viewground came out
identical there (1.01 / 1.00 / 1.00 UniAD L2).

This set makes the trajectory task the RL objective, and pairs each frame with an
evidence view derived from the 3D annotations (data_prep/nuscenes_evidence.py).

Prompts are byte-identical in construction to data_prep/nuscenes_val_prompts.py,
so the policy sees at RL time exactly the format SFT taught and the evaluation
harness sends.

    python data_prep/nuscenes_rl.py \
        --cached_info /mnt/ssd1/vgrl/raw/cached_nuscenes_info.pkl \
        --split_json create_data/full_split.json --split train \
        --nusc_root /nuscenes \
        --evidence_json /mnt/ssd1/vgrl/data/nusc_evidence_train.json \
        --ego_status_json /mnt/ssd1/vgrl/data/ego_status.json \
        --out_json /mnt/ssd1/vgrl/data/nusc_rl.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "create_data"))

from vgrl.prompt_format import (  # noqa: E402
    NUSCENES_INSTRUCTION,
    SYSTEM_NUSCENES,
    VIEW_HEADER,
)
from vgrl.views import CAMERAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_info", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--evidence_json", required=True)
    ap.add_argument("--ego_status_json", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.03)
    ap.add_argument("--require_future", action="store_true", default=True,
                    help="drop frames whose 3 s future is not fully observed; "
                         "their trajectory label is padded and would teach the "
                         "model to emit repeated waypoints")
    ap.add_argument("--require_evidence", action="store_true",
                    help="keep only frames that can form a matched evidence/control "
                         "split (default keeps all; unusable rows still train the "
                         "policy, just without the view term)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--gt_mask_pkl", default=None,
                    help="gt_traj_mask.pkl. Without it the reward scores the padded "
                         "steps of partly-observed futures (18.3% of frames) as if "
                         "they were real targets.")
    args = ap.parse_args()

    from prompt_message import generate_assistant_message, generate_user_message

    data = pickle.load(open(args.cached_info, "rb"))
    evidence = json.load(open(args.evidence_json))
    ego_status = json.load(open(args.ego_status_json)) if args.ego_status_json else {}
    gt_mask = {}
    if args.gt_mask_pkl:
        import numpy as np
        raw = pickle.load(open(args.gt_mask_pkl, "rb"))
        gt_mask = {k: [bool(x) for x in np.asarray(v).reshape(-1, 2)[:, 0] > 0]
                   for k, v in raw.items()}
        print(f"gt_traj_mask: {len(gt_mask)} tokens")

    tokens = json.load(open(args.split_json))[args.split]
    if args.max_samples:
        tokens = tokens[: args.max_samples]
    print(f"{args.split}: {len(tokens)} tokens", flush=True)

    rows, stats = [], Counter()
    for token in tokens:
        info = data.get(token)
        if info is None:
            stats["skip/no_info"] += 1
            continue
        if args.require_future and not bool(info.get("fut_valid_flag", True)):
            stats["skip/future_not_observed"] += 1
            continue

        ev = evidence.get(token)
        if ev is None:
            stats["skip/no_evidence_record"] += 1
            continue
        if args.require_evidence and not ev["vg_usable"]:
            stats["skip/not_vg_usable"] += 1
            continue

        try:
            gt_answer, _stop = generate_assistant_message(data, token, traj_only=True)
            context, images_raw = generate_user_message(data, token)
        except Exception:
            stats["skip/prompt_build_failed"] += 1
            continue

        images, ok = [], True
        for p in images_raw:
            if "/samples/" not in p:
                ok = False
                break
            images.append(os.path.join(args.nusc_root, "samples", p.split("/samples/", 1)[1]))
        if not ok or len(images) != len(CAMERAS):
            stats["skip/bad_images"] += 1
            continue

        ctx = context.strip()
        st = ego_status.get(token)
        if st is not None:
            ctx += (f"\nCurrent longitudinal speed: {st['speed']} m/s"
                    f"\nCurrent longitudinal acceleration: {st['accel']} m/s^2")
        user = VIEW_HEADER + ctx + "\n" + NUSCENES_INSTRUCTION

        rows.append({
            "id": token,
            "images": images,
            # conversational prompt, matching drivelm_rl_v2.json's shape so the
            # trainer and the masking code need no special case
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_NUSCENES}]},
                {"role": "user", "content": (
                    [{"type": "image"} for _ in CAMERAS] + [{"type": "text", "text": user}]
                )},
            ],
            "solution": gt_answer.strip(),
            "category": "trajectory",
            "evidence_views": ev["evidence_views"],
            "vg_usable": bool(ev["vg_usable"]),
            "gt_mask": gt_mask.get(token, [True] * 6),
        })
        stats["kept"] += 1
        stats["kept/vg_usable" if ev["vg_usable"] else "kept/no_view_term"] += 1

    # Split by *scene* would be ideal, but the token list is already MindDriver's
    # official train split; the val slice here is only for monitoring, and the
    # reported numbers come from the held-out nuScenes val split via
    # nuscenes_val_prompts.py. Shuffle with a fixed seed so it is reproducible.
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val, train = rows[:n_val], rows[n_val:]

    base, ext = os.path.splitext(args.out_json)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    for name, part in (("", train), ("_val", val)):
        p = f"{base}{name}{ext}"
        json.dump(part, open(p, "w"))
        print(f"wrote {p}: {len(part)} rows")
    print("stats:", dict(stats))
    if rows:
        print("\n--- first prompt (user turn) ---")
        print(rows[0]["prompt"][1]["content"][-1]["text"][:600])
        print("--- solution ---")
        print(rows[0]["solution"])
        print("--- evidence views ---", [CAMERAS[v] for v in rows[0]["evidence_views"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
