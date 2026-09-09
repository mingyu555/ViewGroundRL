"""Turn DriveLM-nuScenes into an SFT set and an RL set.

Input is DriveLM's `v1_1_train_nus.json`:

    {scene_token: {"key_frames": {frame_token: {
        "QA": {"perception"|"prediction"|"planning"|"behavior": [{"Q","A",...}]},
        "image_paths": {"CAM_FRONT": "../nuscenes/samples/...", ...},
        "key_object_infos": {...}}}}}

Two outputs:

  --out_sft   LLaMA-Factory sharegpt format, for the plain SFT stage.
  --out_rl    TRL GRPO format: one row per QA with `prompt` (conversational),
              `images` (the six views in vgrl.views.CAMERAS order), `solution`
              (reference answer) and `evidence_views` — the view indices the
              answer's `<c1,CAM_BACK,...>` tags point at, which is what the
              view-grounding term masks.

Rows whose evidence views cannot supply a matched control split (no camera named,
or all six named) are kept but flagged: the trainer scores them with plain GRPO
and skips the perception term, so they still contribute to the policy gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.prompt_format import (  # noqa: E402
    SYSTEM_DRIVELM,
    SYSTEM_DRIVELM_PLAIN,
    VIEW_HEADER,
)
from vgrl.views import CAMERAS, evidence_views  # noqa: E402


def resolve_paths(image_paths: dict, nusc_root: str) -> list[str] | None:
    """DriveLM stores '../nuscenes/samples/...'; re-root onto the local copy."""
    out = []
    for cam in CAMERAS:
        rel = image_paths.get(cam)
        if not rel:
            return None
        marker = "/samples/"
        if marker not in rel:
            return None
        out.append(os.path.join(nusc_root, "samples", rel.split(marker, 1)[1]))
    return out


def iter_qa(data: dict):
    for scene_token, scene in data.items():
        for frame_token, frame in scene.get("key_frames", {}).items():
            image_paths = frame.get("image_paths") or {}
            for category, items in (frame.get("QA") or {}).items():
                for qa_i, qa in enumerate(items or []):
                    q, a = qa.get("Q"), qa.get("A")
                    if not q or not a:
                        continue
                    yield {
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "category": category,
                        "qa_index": qa_i,
                        "question": q,
                        "answer": a,
                        "image_paths": image_paths,
                    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivelm_json", required=True)
    ap.add_argument("--nusc_root", required=True, help="local nuScenes root (has samples/)")
    ap.add_argument("--out_sft", default=None)
    ap.add_argument("--out_rl", default=None)
    ap.add_argument("--answer_view_source", default="both", choices=["answer", "question", "both"])
    ap.add_argument("--categories", default="perception,prediction,planning,behavior")
    ap.add_argument("--rl_categories", default="perception,prediction,planning",
                    help="RL is most meaningful where the answer cites objects; "
                         "'behavior' rarely names a camera.")
    ap.add_argument("--require_evidence_for_rl", action="store_true",
                    help="drop RL rows that cannot supply a matched evidence/control split")
    ap.add_argument("--check_images", action="store_true")
    ap.add_argument("--val_fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep = set(args.categories.split(","))
    rl_keep = set(args.rl_categories.split(","))
    data = json.load(open(args.drivelm_json))
    print(f"scenes: {len(data)}")

    sft_rows, rl_rows = [], []
    stats = Counter()
    missing_images = 0

    for rec in iter_qa(data):
        stats[f"qa/{rec['category']}"] += 1
        if rec["category"] not in keep:
            continue

        images = resolve_paths(rec["image_paths"], args.nusc_root)
        if images is None:
            missing_images += 1
            stats["skip/no_image_paths"] += 1
            continue
        if args.check_images and not all(os.path.exists(p) for p in images):
            missing_images += 1
            stats["skip/image_absent"] += 1
            continue

        ev = evidence_views(rec["question"], rec["answer"], source=args.answer_view_source)
        stats[f"evidence_views/{len(ev)}"] += 1
        # a matched control split needs at least one evidence view and one non-evidence view
        usable = 0 < len(ev) < len(CAMERAS)

        sample_id = f"{rec['frame_token']}::{rec['category']}::{rec['qa_index']}"
        user_text = VIEW_HEADER + rec["question"]

        sft_rows.append(
            {
                "id": sample_id,
                "images": images,
                "system": SYSTEM_DRIVELM_PLAIN,
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": rec["answer"]},
                ],
            }
        )

        if rec["category"] in rl_keep:
            if args.require_evidence_for_rl and not usable:
                stats["skip/rl_no_matched_split"] += 1
            else:
                rl_rows.append(
                    {
                        "id": sample_id,
                        "prompt": [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_DRIVELM}]},
                            {
                                "role": "user",
                                "content": [{"type": "image"} for _ in CAMERAS]
                                + [{"type": "text", "text": rec["question"]}],
                            },
                        ],
                        "images": images,
                        "solution": rec["answer"],
                        "category": rec["category"],
                        "evidence_views": ev,
                        "vg_usable": usable,
                    }
                )
                stats["rl/usable" if usable else "rl/no_matched_split"] += 1

    rng = random.Random(args.seed)

    def split_and_write(rows, path):
        if not path or not rows:
            return
        rng.shuffle(rows)
        n_val = int(len(rows) * args.val_fraction)
        val, train = rows[:n_val], rows[n_val:]
        base, ext = os.path.splitext(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        for name, part in (("", train), ("_val", val)):
            if not part:
                continue
            out = f"{base}{name}{ext}"
            with open(out, "w") as f:
                json.dump(part, f, ensure_ascii=False)
            print(f"wrote {out}: {len(part)} rows")

    split_and_write(sft_rows, args.out_sft)
    split_and_write(rl_rows, args.out_rl)

    print("\n--- stats ---")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if missing_images:
        print(f"  (images unresolved for {missing_images} QA pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
