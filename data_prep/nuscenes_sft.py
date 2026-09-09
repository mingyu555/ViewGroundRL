"""nuScenes planning data for the SFT stage, in the same sharegpt shape as the
DriveLM SFT file so the two can simply be concatenated.

This reuses MindDriver's own prompt builders (create_data/prompt_message.py) for
the ego history / mission-goal / trajectory formatting, but drops the MoVQGAN
future-image tokens: the SFT stage here is plain text-answer supervision, and the
view-grounding RL stage that follows operates on the six current views only.

Requires DriveLM's cached info pickle (`cached_nuscenes_info.pkl`, the same file
MindDriver uses) and the nuScenes split json.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "create_data"))

from vgrl.prompt_format import (  # noqa: E402
    NUSCENES_INSTRUCTION as INSTRUCTION,
    SYSTEM_NUSCENES_PLAIN as SYSTEM,
    VIEW_HEADER,
)
from vgrl.views import CAMERAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_info", required=True, help="cached_nuscenes_info.pkl")
    ap.add_argument("--split_json", required=True, help="full_split.json with train/val token lists")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--check_images", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from prompt_message import generate_assistant_message, generate_user_message  # noqa: E402

    data = pickle.load(open(args.cached_info, "rb"))
    tokens = json.load(open(args.split_json))[args.split]
    if args.max_samples:
        tokens = tokens[: args.max_samples]
    print(f"{args.split}: {len(tokens)} tokens")

    rows, skipped = [], 0
    for token in tokens:
        if token not in data:
            skipped += 1
            continue
        try:
            assistant, _stop = generate_assistant_message(data, token, traj_only=True)
            user, images_rel = generate_user_message(data, token)
        except Exception:
            skipped += 1
            continue

        # generate_user_message returns paths rooted at the authors' layout; re-root
        images = []
        ok = True
        for p in images_rel:
            marker = "/samples/"
            if marker not in p:
                ok = False
                break
            images.append(os.path.join(args.nusc_root, "samples", p.split(marker, 1)[1]))
        if not ok or len(images) != len(CAMERAS):
            skipped += 1
            continue
        if args.check_images and not all(os.path.exists(p) for p in images):
            skipped += 1
            continue

        rows.append(
            {
                "id": token,
                "images": images,
                "system": SYSTEM,
                "conversations": [
                    {"from": "human", "value": VIEW_HEADER + user + INSTRUCTION},
                    {"from": "gpt", "value": assistant},
                ],
            }
        )

    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"wrote {args.out_json}: {len(rows)} rows ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
