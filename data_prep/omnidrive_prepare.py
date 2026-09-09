"""OmniDrive vqa 를 우리 학습/평가 형식으로 바꾼다. 근거뷰는 3D 좌표를 카메라에 투영해 얻는다.

OmniDrive 정답은 카메라 이름을 쓰지 않고 ego 좌표를 쓴다:
    "there is a moving car behind us ... located at (-27.4, +2.4) with a velocity of (+7.8, -1.6)"
규약은 x=전방, y=좌측 (실측 확정: frontleft 254건이 x>0 99% / y>0 98%).

좌표 정확도 검증: 정답 좌표를 pkl 의 gt_boxes 와 매칭했을 때 최근접 L1 거리
중앙값이 0.06m, 82.7% 가 1m 이내다. GPT-4 로 생성했지만 좌표는 실제 nuScenes
어노테이션에서 온 것이다.

근거뷰 결정: 좌표를 gt_boxes 에 매칭해 z 를 얻고(정답에는 x,y 만 있다), 각 카메라의
sensor2ego 역변환 + cam_intrinsic 으로 투영해 화면 안에 들어오는 카메라를 모은다.
매칭이 안 되는 좌표는 z=1.0(전형적 차량 중심 높이)으로 가정한다.

질문 힌트 누출이 거의 없다는 점이 이 데이터셋의 핵심 장점이다 (방향어 0.1~0.7%).
NuInstruct 는 54% 가 질문에서 카메라를 알려줘 gap 이 자명하게 커졌다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COORD = re.compile(r"\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)")
PT = re.compile(r"\[PT[,\s]")
IMG_W, IMG_H = 1600, 900
# 우리가 모델에 넣는 뷰 순서 (인덱스 = 이 리스트의 위치)
VIEW_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
              "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

SYSTEM = (
    "You are the perception and reasoning module of an autonomous vehicle. You are "
    "given the six surround-view camera images of the current frame, in this order: "
    + ", ".join(VIEW_ORDER) + ". Object locations are given in ego coordinates in "
    "meters, where +x is forward and +y is to the left. Answer the question."
)


def quat_to_rot(q):
    """[w, x, y, z] -> 3x3 회전행렬."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def cam_projectors(info):
    """카메라별 (R_ego2sensor, t, K). 없는 카메라는 건너뛴다."""
    out = {}
    for name in VIEW_ORDER:
        c = info["cams"].get(name)
        if c is None:
            continue
        R = quat_to_rot(np.asarray(c["sensor2ego_rotation"], dtype=float))
        t = np.asarray(c["sensor2ego_translation"], dtype=float)
        K = np.asarray(c["cam_intrinsic"], dtype=float)
        out[name] = (R.T, t, K)          # R.T 로 ego->sensor
    return out


def visible_views(pt_ego, proj):
    """ego 좌표 점이 보이는 카메라 인덱스 집합."""
    out = set()
    for i, name in enumerate(VIEW_ORDER):
        if name not in proj:
            continue
        Rt, t, K = proj[name]
        p = Rt @ (np.asarray(pt_ego, dtype=float) - t)
        if p[2] <= 0.5:                  # 카메라 뒤 또는 너무 가까움
            continue
        uv = K @ p
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            out.add(i)
    return out


class _Stub:
    """shapely 자리표시자. 언피클만 통과시키고 내용은 버린다."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _TolerantUnpickler(pickle.Unpickler):
    """shapely 없이 nuScenes infos pkl 을 읽는다.

    pkl 의 map_geoms 에 shapely 기하 객체가 들어 있어 shapely 가 없으면 언피클이
    ModuleNotFoundError 로 죽는다. 우리가 쓰는 필드는 token / cams / gt_boxes /
    ego2global 뿐이라 shapely 클래스는 자리표시자로 바꿔도 무해하다.
    """

    def find_class(self, module, name):
        if module.split(".")[0] == "shapely":
            return _Stub
        return super().find_class(module, name)


def _load_infos(path: str):
    with open(path, "rb") as fh:
        return _TolerantUnpickler(fh).load()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa_dir", required=True, help="vqa/train 또는 vqa/val")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--nusc_root", default="/nuscenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit_frames", type=int, default=0)
    ap.add_argument("--match_thresh", type=float, default=2.0,
                    help="gt_boxes 매칭 L1 임계(m). 넘으면 z=1.0 가정")
    ap.add_argument("--check_images", action="store_true")
    args = ap.parse_args()

    d = _load_infos(args.pkl)
    infos = {x["token"]: x for x in d["infos"]}
    files = sorted(glob.glob(os.path.join(args.vqa_dir, "*.json")))
    if args.limit_frames:
        files = files[: args.limit_frames]

    rows, stats = [], Counter()
    nview_hist = Counter()
    for f in files:
        tok = os.path.basename(f)[:-5]
        info = infos.get(tok)
        if info is None:
            stats["skip/no_frame_info"] += 1
            continue
        try:
            images = [os.path.join(args.nusc_root,
                                   re.sub(r"^.*?samples/", "samples/",
                                          info["cams"][c]["data_path"]))
                      for c in VIEW_ORDER]
        except KeyError:
            stats["skip/missing_cam"] += 1
            continue
        if args.check_images and not all(os.path.exists(p) for p in images):
            stats["skip/image_absent"] += 1
            continue

        proj = cam_projectors(info)
        gb = np.asarray(info.get("gt_boxes"))
        try:
            qas = json.load(open(f))
        except Exception:
            stats["skip/bad_json"] += 1
            continue

        for qi, qa in enumerate(qas if isinstance(qas, list) else []):
            if not isinstance(qa, dict) or "question" not in qa:
                continue
            q, a = qa["question"], qa["answer"]
            is_cf = bool(PT.search(q) or re.search(r"if you follow|risks if", q.lower()))
            # 질문의 궤적 좌표는 근거가 아니다. 정답 좌표만 쓴다.
            ev = set()
            n_matched = 0
            for mx, my in COORD.findall(a):
                x, y = float(mx), float(my)
                z = 1.0
                if gb.size:
                    dd = np.abs(gb[:, 0] - x) + np.abs(gb[:, 1] - y)
                    j = int(dd.argmin())
                    if dd[j] <= args.match_thresh:
                        z = float(gb[j, 2])
                        n_matched += 1
                ev |= visible_views((x, y, z), proj)
            ev = sorted(ev)
            nview_hist[len(ev)] += 1

            rows.append({
                "id": f"omni::{tok}::{qi}",
                "frame_token": tok,
                "task": "omni-counterfactual" if is_cf else "omni-vqa",
                "category": "counterfactual" if is_cf else "vqa",
                "question": q,
                "solution": a,
                "images": images,
                "system": SYSTEM,
                "conversations": [
                    {"from": "human", "value": "<image>" * len(VIEW_ORDER) + q},
                    {"from": "gpt", "value": a},
                ],
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "image"} for _ in VIEW_ORDER]
                     + [{"type": "text", "text": q}]},
                ],
                "evidence_views": ev,
                "vg_usable": 0 < len(ev) < len(VIEW_ORDER),
                "n_coord_matched": n_matched,
            })
            stats[f"kept/{rows[-1]['task']}"] += 1

    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    u = sum(1 for r in rows if r["vg_usable"])
    print(f"총 {len(rows)}행 -> {args.out}")
    print(f"vg_usable: {u} ({100*u/max(len(rows),1):.1f}%)")
    print("근거뷰 개수 분포:", dict(sorted(nview_hist.items())))
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
