"""Ego velocity and acceleration per nuScenes sample, straight from the CAN bus.

MindDriver's prompt hands the model `速度 3.3 m/s` and `加速度 0.6 m/s^2` as numbers.
Ours only gave four past waypoints, from which the model had to infer both — and a
diagnostic showed our SFT model landing at UniAD L2 1.50 against 1.51 for a plain
constant-acceleration fit of those waypoints, i.e. no gain from the images at all.
Feeding the same explicit ego state is what makes the comparison fair.

Reproduces MindDriver's extraction (gen_data/convert_to_qwen_img.py) without
nuscenes-devkit: for each sample, take the `vehicle_monitor` message nearest the
LIDAR_TOP timestamp for speed (km/h -> m/s) and the `pose` message nearest it for
longitudinal acceleration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left


def nearest(sorted_times: list[int], records: list[dict], t: int) -> dict | None:
    if not sorted_times:
        return None
    i = bisect_left(sorted_times, t)
    best, best_d = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sorted_times):
            d = abs(sorted_times[j] - t)
            if best_d is None or d < best_d:
                best, best_d = records[j], d
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="nuScenes v1.0-trainval dir")
    ap.add_argument("--can_bus", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    print("reading sample.json / scene.json ...", flush=True)
    samples = json.load(open(f"{args.tables}/sample.json"))
    scenes = {s["token"]: s["name"] for s in json.load(open(f"{args.tables}/scene.json"))}

    print("reading sample_data.json for LIDAR_TOP timestamps ...", flush=True)
    lidar_ts: dict[str, int] = {}
    for r in json.load(open(f"{args.tables}/sample_data.json")):
        if r["is_key_frame"] and r["filename"].startswith("samples/LIDAR_TOP"):
            lidar_ts[r["sample_token"]] = r["timestamp"]
    print(f"  LIDAR_TOP keyframes: {len(lidar_ts)}", flush=True)

    # group samples by scene so each scene's CAN files are read once
    by_scene: dict[str, list[dict]] = {}
    for s in samples:
        by_scene.setdefault(s["scene_token"], []).append(s)

    out: dict[str, dict] = {}
    missing_can = skipped = 0
    for scene_token, scene_samples in by_scene.items():
        name = scenes.get(scene_token)
        vm_path = f"{args.can_bus}/{name}_vehicle_monitor.json"
        pose_path = f"{args.can_bus}/{name}_pose.json"
        if not (name and os.path.exists(vm_path) and os.path.exists(pose_path)):
            # nuScenes ships no CAN bus for a handful of scenes
            missing_can += len(scene_samples)
            continue

        vm = json.load(open(vm_path))
        pose = json.load(open(pose_path))
        vm_t = [m["utime"] for m in vm]
        pose_t = [m["utime"] for m in pose]

        for s in scene_samples:
            t = lidar_ts.get(s["token"])
            if t is None:
                skipped += 1
                continue
            m = nearest(vm_t, vm, t)
            p = nearest(pose_t, pose, t)
            if m is None or p is None:
                skipped += 1
                continue
            out[s["token"]] = {
                "speed": round(m.get("vehicle_speed", 0.0) / 3.6, 2),   # km/h -> m/s
                "accel": round((p.get("accel") or [0.0])[0], 2),        # longitudinal
            }

    print(f"ego status for {len(out)} samples "
          f"({missing_can} in scenes without CAN bus, {skipped} otherwise skipped)")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out_json}")
    if out:
        k = next(iter(out))
        print(f"  example {k}: {out[k]}")
        speeds = sorted(v["speed"] for v in out.values())
        print(f"  speed m/s: min={speeds[0]} median={speeds[len(speeds)//2]} max={speeds[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
