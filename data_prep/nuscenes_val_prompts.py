"""Build the nuScenes val prompt set in *our* SFT format, for L2 / collision eval.

The MindDriver val json cannot be reused directly: its prompts are Chinese, carry
CAN-bus speed/acceleration, and ask for MoVQGAN future-image tokens. This emits the
same 6-view planning question in the format the SFT stage taught
(`SYSTEM_NUSCENES` + `VIEW_HEADER` + ego history + `NUSCENES_INSTRUCTION`,
answered as `<think>...</think><answer>[(x,y) x6]</answer>`).

Ground truth for scoring stays exactly MindDriver's — the same `gt_traj.pkl`,
`gt_traj_mask.pkl` and occupancy maps — so the resulting numbers are directly
comparable to the reproduction figures.

One difference to keep in mind when comparing: our prompt gives the four past
waypoints at 0.5 s spacing but not explicit velocity/acceleration. Those four
points imply both, so this sits on the paper's "with ego status" side, but it is
not the identical input MindDriver received.

Output shape matches infer_vllm.py's expectations: a list of
{id, images, messages:[user, assistant]}.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "create_data"))

from vgrl.prompt_format import (  # noqa: E402
    NUSCENES_INSTRUCTION,
    SYSTEM_NUSCENES,
    VIEW_HEADER,
    wrap,
)
from vgrl.views import CAMERAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_info", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--check_images", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--ego_status_json", default=None,
                    help="must match what the SFT prompts carried, or the model sees a "
                         "different input format at eval time than it trained on")
    args = ap.parse_args()

    from prompt_message import generate_assistant_message, generate_user_message

    ego_status = json.load(open(args.ego_status_json)) if args.ego_status_json else {}

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
            gt_answer, _stop = generate_assistant_message(data, token, traj_only=True)
            context, images_raw = generate_user_message(data, token)
        except Exception:
            skipped += 1
            continue

        images, ok = [], True
        for p in images_raw:
            if "/samples/" not in p:
                ok = False
                break
            images.append(os.path.join(args.nusc_root, "samples", p.split("/samples/", 1)[1]))
        if not ok or len(images) != len(CAMERAS):
            skipped += 1
            continue
        if args.check_images and not all(os.path.exists(p) for p in images):
            skipped += 1
            continue

        ctx = context.strip()
        ego = ego_status.get(token)
        if ego is not None:
            ctx += (f"\nCurrent longitudinal speed: {ego['speed']} m/s"
                    f"\nCurrent longitudinal acceleration: {ego['accel']} m/s^2")
        # No future-validity filter here: evaluation must cover the whole val split,
        # and the metric masks the unobserved steps itself.
        user = VIEW_HEADER + ctx + "\n" + NUSCENES_INSTRUCTION
        rows.append(
            {
                "id": token,
                "images": images,
                "messages": [
                    {"role": "user", "content": SYSTEM_NUSCENES + "\n\n" + user},
                    # reference target, only used as the `label` column in the
                    # prediction jsonl — scoring uses the GT pickles
                    {"role": "assistant", "content": wrap("", gt_answer.strip())},
                ],
            }
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"wrote {args.out_json}: {len(rows)} rows ({skipped} skipped)")
    if rows:
        print("\n--- first prompt ---")
        print(rows[0]["messages"][0]["content"][:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
