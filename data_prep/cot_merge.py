"""Merge per-dataset CoT SFT files into one training set, with a held-out split.

Splitting is done **by frame token**, not by row: nuScenes and DriveLM both give
several rows per frame (six QA pairs, or a planning sample sharing the frame), and
a random row split would put the same images on both sides of the boundary and
inflate the validation numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter


def frame_key(row: dict) -> str:
    """Group rows that share a frame so they land on the same side of the split."""
    uid = row.get("id", "")
    if uid.startswith("drivelm::"):
        parts = uid.split("::")
        return f"frame::{parts[1]}" if len(parts) > 1 else uid
    if uid.startswith("nusc::"):
        return f"frame::{uid.split('::', 1)[1]}"
    return uid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.01)
    ap.add_argument("--max_per_task", type=int, default=0,
                    help="cap rows per task, to balance a large set against a small one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_task: dict[str, list[dict]] = {}
    stats = Counter()

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"  skipping missing {path}")
            continue
        rows = json.load(open(path))
        for r in rows:
            task = r.get("task", "unknown")
            by_task.setdefault(task, []).append(r)
            stats[f"in/{task}"] += 1
            stats[f"in/{task}/pass{r.get('cot_pass', '?')}"] += 1
        print(f"  {path}: {len(rows)} rows")

    merged: list[dict] = []
    for task, rows in by_task.items():
        rng.shuffle(rows)
        if args.max_per_task and len(rows) > args.max_per_task:
            stats[f"capped/{task}"] = len(rows) - args.max_per_task
            rows = rows[: args.max_per_task]
        merged.extend(rows)

    if not merged:
        raise SystemExit("no rows to merge")

    frames = sorted({frame_key(r) for r in merged})
    rng.shuffle(frames)
    n_val_frames = max(1, int(len(frames) * args.val_fraction)) if args.val_fraction else 0
    val_frames = set(frames[:n_val_frames])

    train = [r for r in merged if frame_key(r) not in val_frames]
    val = [r for r in merged if frame_key(r) in val_frames]

    base, ext = os.path.splitext(args.out_json)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    for name, part in (("", train), ("_val", val)):
        if not part:
            continue
        out = f"{base}{name}{ext}"
        with open(out, "w") as f:
            json.dump(part, f, ensure_ascii=False)
        print(f"wrote {out}: {len(part)} rows")

    print("--- stats ---")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  frames: {len(frames)} total, {len(val_frames)} held out")
    print(f"  train/val rows: {len(train)}/{len(val)}")
    overlap = {frame_key(r) for r in train} & {frame_key(r) for r in val}
    print(f"  frame overlap between splits: {len(overlap)} (must be 0)")
    if overlap:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
