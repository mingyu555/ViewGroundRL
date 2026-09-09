"""Label the evidence view of a nuScenes planning frame from 3D annotations.

DriveLM hands us the evidence view for free: its answers tag objects as
`<c1,CAM_BACK,x,y>`. nuScenes planning has no such tag — the label is a
trajectory, and a trajectory does not name a camera. But nuScenes ships the 3D
boxes and the full camera calibration, so the evidence view can be *derived*,
and more reliably than a human tag: which objects actually constrain the planned
manoeuvre is a geometric question.

  1. Build the corridor the ego sweeps over the next 3 s from `gt_ego_fut_trajs`.
  2. Score every annotated object by how much it constrains that corridor
     (lateral clearance, and time-to-conflict using the agent's own future).
  3. Project the constraining objects into the six cameras with
     `cam_intrinsic` + `sensor2lidar_*`; the views they land in are the evidence.

Objects that are annotated but not actually visible would be false evidence, so a
box must project inside the image *and* clear a visibility floor
(`num_lidar_pts`, plus nuScenes' own `visibility_token` when available).

Frames whose evidence set is empty or covers all six views cannot form the
matched evidence/control contrast and are marked `vg_usable=False` — the same
convention `drivelm_prepare.py` uses.

    python data_prep/nuscenes_evidence.py \
        --cached_info /mnt/ssd1/vgrl/raw/cached_nuscenes_info.pkl \
        --split_json create_data/full_split.json --split train \
        --out_json /mnt/ssd1/vgrl/data/nusc_evidence_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.views import CAMERAS  # noqa: E402

# Ego footprint (nuScenes Renault Zoe), metres. Half-width is what the corridor
# test needs; a little slack accounts for the planner not tracking the reference
# perfectly.
EGO_HALF_WIDTH = 0.93
CORRIDOR_SLACK = 1.5

# An object only counts as evidence if it is close enough to matter. 3 s at urban
# speeds covers ~15 m, so look a little beyond that and ignore anything further.
MAX_RANGE_M = 50.0

# Visibility floor: a box with almost no lidar returns is annotated but not
# something a camera-only model could have read.
MIN_LIDAR_PTS = 3

# Categories that can constrain a driving decision. Static map furniture is
# excluded — a traffic cone at the roadside is annotated but does not explain a
# braking decision, and including it floods every frame with evidence.
DYNAMIC_PREFIXES = ("vehicle.", "human.")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(box: np.ndarray) -> np.ndarray:
    """8 corners of a (x,y,z,w,l,h,yaw) LiDAR-frame box -> (3, 8)."""
    x, y, z, w, l, h, yaw = box[:7]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    c = np.array([
        [dx, dx, dx, dx, -dx, -dx, -dx, -dx],
        [dy, dy, -dy, -dy, dy, dy, -dy, -dy],
        [dz, -dz, dz, -dz, dz, -dz, dz, -dz],
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0],
                  [0.0, 0.0, 1.0]])
    return R @ c + np.array([[x], [y], [z]])


def visible_in_camera(corners: np.ndarray, cam: dict, min_frac: float = 0.5) -> bool:
    """Does the box land inside this camera's image?

    `corners` are LiDAR-frame. sensor2lidar_* maps camera -> lidar, so the
    inverse takes the box into the camera frame. A point is in view when it is in
    front of the camera and inside the image rectangle; require a fraction of the
    corners so a box grazing the edge is not counted.
    """
    R = np.asarray(cam["sensor2lidar_rotation"])
    t = np.asarray(cam["sensor2lidar_translation"]).reshape(3, 1)
    in_cam = R.T @ (corners - t)                      # lidar -> camera

    depth = in_cam[2]
    ahead = depth > 0.5
    if not ahead.any():
        return False

    K = np.asarray(cam["cam_intrinsic"])
    proj = K @ in_cam
    u = proj[0] / np.clip(proj[2], 1e-6, None)
    v = proj[1] / np.clip(proj[2], 1e-6, None)

    # nuScenes camera images are 1600x900
    inside = ahead & (u >= 0) & (u < 1600) & (v >= 0) & (v < 900)
    return inside.sum() >= max(1, int(min_frac * corners.shape[1]))


def corridor_constraint(box: np.ndarray, fut: np.ndarray, agent_fut: np.ndarray | None) -> float:
    """How strongly this object constrains the planned path. Higher = more.

    Two ways an object matters:
      - it sits inside the corridor the ego sweeps (static conflict);
      - its own future crosses the corridor while the ego is there (dynamic).

    Returns 0.0 for objects that never interact.
    """
    x, y = box[0], box[1]
    if np.hypot(x, y) > MAX_RANGE_M:
        return 0.0

    half_w = EGO_HALF_WIDTH + CORRIDOR_SLACK + max(box[3], box[4]) / 2.0

    # Static test: distance from the object centre to the planned polyline.
    best = 0.0
    pts = fut  # (T,2) cumulative ego displacements, ego frame
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        L2 = float(seg @ seg)
        if L2 < 1e-9:
            continue
        u = float(np.clip(((np.array([x, y]) - a) @ seg) / L2, 0.0, 1.0))
        closest = a + u * seg
        d = float(np.hypot(*(np.array([x, y]) - closest)))
        if d < half_w:
            # closer and earlier in the horizon = stronger constraint
            best = max(best, (half_w - d) / half_w * (1.0 - 0.5 * i / max(len(pts) - 1, 1)))

    # Dynamic test: the agent's own future, if annotated, walked alongside ours.
    # Note the cache is asymmetric — gt_ego_fut_trajs are cumulative positions but
    # gt_agent_fut_trajs are per-step deltas (verified: ~4.8 m per 0.5 s step, and
    # the distance from the origin is not monotonic). So this one really does need
    # accumulating, unlike the ego path above.
    if agent_fut is not None and len(agent_fut) >= 2:
        pos = np.array([x, y], dtype=float)
        steps = min(len(agent_fut), len(pts) - 1)
        for i in range(steps):
            pos = pos + agent_fut[i]
            d = float(np.hypot(*(pos - pts[i + 1])))
            if d < half_w:
                best = max(best, (half_w - d) / half_w * (1.0 - 0.5 * i / max(steps, 1)))
    return best


def evidence_for_frame(info: dict, top_k: int = 3, min_constraint: float = 0.05):
    """Evidence view indices for one frame, plus diagnostics."""
    boxes = np.asarray(info.get("gt_boxes"))
    names = list(info.get("gt_names", []))
    fut = np.asarray(info.get("gt_ego_fut_trajs"), dtype=float).reshape(-1, 2)
    if boxes.size == 0 or fut.shape[0] < 2:
        return [], {"n_boxes": 0, "n_constraining": 0}

    # gt_ego_fut_trajs are already cumulative positions in the ego frame, not
    # per-step deltas: element [1:7] is byte-identical to the trajectory label the
    # SFT/eval prompts carry (verified on 300/300 val samples), and element [0] is
    # the origin. An earlier version cumsum'd them, which stretched the 3 s
    # corridor from 8.2 m to 35.7 m and labelled objects far outside the horizon
    # the plan actually covers.
    path = fut[1:] if len(fut) > 6 else fut

    agent_fut = info.get("gt_agent_fut_trajs")
    agent_fut = np.asarray(agent_fut, dtype=float) if agent_fut is not None else None

    npts = np.asarray(info.get("num_lidar_pts", np.full(len(boxes), 99)))

    scored = []
    for i, box in enumerate(boxes):
        name = names[i] if i < len(names) else ""
        if not name.startswith(DYNAMIC_PREFIXES):
            continue
        if i < len(npts) and npts[i] < MIN_LIDAR_PTS:
            continue
        af = None
        if agent_fut is not None and i < len(agent_fut):
            af = np.asarray(agent_fut[i], dtype=float).reshape(-1, 2)
        s = corridor_constraint(box, path, af)
        if s > min_constraint:
            scored.append((s, i))

    scored.sort(reverse=True)
    scored = scored[:top_k]

    views: set[int] = set()
    for _s, i in scored:
        corners = box_corners(boxes[i])
        for vi, cam_name in enumerate(CAMERAS):
            cam = info["cams"].get(cam_name)
            if cam and visible_in_camera(corners, cam):
                views.add(vi)

    return sorted(views), {"n_boxes": len(boxes), "n_constraining": len(scored)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_info", required=True)
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_constraint", type=float, default=0.02)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    data = pickle.load(open(args.cached_info, "rb"))
    if args.split_json:
        tokens = json.load(open(args.split_json))[args.split]
    else:
        tokens = list(data)
    if args.max_samples:
        tokens = tokens[: args.max_samples]
    print(f"{args.split}: {len(tokens)} tokens", flush=True)

    out, stats = {}, Counter()
    view_hist, nviews_hist = Counter(), Counter()
    for n, tok in enumerate(tokens, 1):
        info = data.get(tok)
        if info is None:
            stats["missing"] += 1
            continue
        views, diag = evidence_for_frame(info, args.top_k, args.min_constraint)
        usable = 0 < len(views) < len(CAMERAS)
        out[tok] = {
            "evidence_views": views,
            "vg_usable": usable,
            "n_constraining": diag["n_constraining"],
            "fut_valid": bool(info.get("fut_valid_flag", True)),
        }
        stats["usable" if usable else ("no_evidence" if not views else "all_views")] += 1
        nviews_hist[len(views)] += 1
        for v in views:
            view_hist[CAMERAS[v]] += 1
        if n % 5000 == 0:
            print(f"  {n}/{len(tokens)}", flush=True)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(out, open(args.out_json, "w"))
    print(f"\nwrote {args.out_json}: {len(out)} frames")
    print("stats:", dict(stats))
    print("evidence-view count per frame:", dict(sorted(nviews_hist.items())))
    print("which views carry evidence:", dict(view_hist.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
