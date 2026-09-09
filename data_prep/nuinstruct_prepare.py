"""NuInstruct 를 우리 학습/평가 형식으로 바꾼다. 현재 프레임 6뷰만 쓴다.

원본은 비디오(3~5 프레임 x 6뷰)지만 VGGDrive 논문이 같은 벤치마크를 이렇게 쓴다:
  "For benchmarks constructed from the nuScenes dataset (NuInstruct, DriveLM,
   OmniDrive, nuScenes-Plan), we use six surround-view images from the current
   frame (C=6)."
img_paths 의 타임스탬프가 증가하므로 마지막 원소가 현재 프레임이다.

카메라 인덱스 규약: img_paths dict 의 키 순서 그대로다.
  c0=CAM_FRONT_LEFT c1=CAM_FRONT c2=CAM_FRONT_RIGHT
  c3=CAM_BACK_LEFT  c4=CAM_BACK  c5=CAM_BACK_RIGHT
("front_right 의 가장 가까운 객체" 질문의 정답이 c2 인 것으로 검증했다.)

근거뷰 라벨은 정답의 <class>[cN,...] 에서 뽑는다. task 마다 비율이 크게 다르다:
  100%  perception-closest, risk-{approaching,crossing,lane_change,on_coming,
        braking,overtaking}
  ~30%  reasoning, perception-in_the_same_road
  0%    나머지 (수치/상태 질문이라 객체를 지목하지 않는다)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAM_RE = re.compile(r"\[c(\d+)\s*,")
# NuInstruct 의 6뷰 순서 (img_paths dict 키 순서와 동일)
NI_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
              "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

SYSTEM = (
    "You are the perception and reasoning module of an autonomous vehicle. You are "
    "given the six surround-view camera images of the current frame, in this order: "
    + ", ".join(NI_CAMERAS) + ". The n-th camera is referred to as cn, so c0 is "
    + NI_CAMERAS[0] + " and c5 is " + NI_CAMERAS[5] + ". Answer the question using "
    "exactly the format the question asks for."
)


def evidence_views(answer: str) -> list[int]:
    return sorted({int(c) for c in CAM_RE.findall(answer or "") if 0 <= int(c) < 6})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="NuInstruct train.json / val.json")
    ap.add_argument("--nusc_root", default="/nuscenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_task", type=int, default=0, help="task 당 상한 (0=전부)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check_images", action="store_true")
    args = ap.parse_args()

    src = json.load(open(args.ann))
    by_task = defaultdict(list)
    stats = Counter()

    for r in src:
        frames = r["img_paths"]
        if not frames:
            stats["skip/no_frames"] += 1
            continue
        cur = frames[-1]                     # 타임스탬프 증가 → 마지막이 현재 프레임
        try:
            images = [os.path.join(args.nusc_root, cur[c]) for c in NI_CAMERAS]
        except KeyError as e:
            stats[f"skip/missing_{e}"] += 1
            continue
        if args.check_images and not all(os.path.exists(p) for p in images):
            stats["skip/image_absent"] += 1
            continue

        ev = evidence_views(r["Answer"])
        by_task[r["task"]].append({
            "id": f"nuins::{r['task']}::{r['qa_id']}",
            "task": r["task"],
            "category": r["task"].split("-")[0],       # perception/prediction/risk/reasoning
            "question": r["Question"],
            "solution": r["Answer"],
            "images": images,
            "system": SYSTEM,
            "conversations": [
                {"from": "human", "value": "<image>" * len(NI_CAMERAS) + r["Question"]},
                {"from": "gpt", "value": r["Answer"]},
            ],
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"} for _ in NI_CAMERAS]
                 + [{"type": "text", "text": r["Question"]}]},
            ],
            "evidence_views": ev,
            "vg_usable": 0 < len(ev) < len(NI_CAMERAS),
            "n_frames_original": len(frames),
        })
        stats[f"kept/{r['task']}"] += 1

    rng = random.Random(args.seed)
    rows = []
    print(f"{'task':30s}{'가용':>8s}{'선택':>8s}{'vg_usable':>11s}")
    for task in sorted(by_task):
        v = by_task[task]
        rng.shuffle(v)
        take = v[: args.per_task] if args.per_task else v
        rows += take
        u = sum(1 for r in take if r["vg_usable"])
        print(f"{task:30s}{len(v):8d}{len(take):8d}{100*u/max(len(take),1):10.1f}%")

    rng.shuffle(rows)
    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    tot_u = sum(1 for r in rows if r["vg_usable"])
    print(f"\n총 {len(rows)}행 -> {args.out}")
    print(f"vg_usable(근거뷰 라벨 있음): {tot_u} ({100*tot_u/len(rows):.1f}%)")
    print("카테고리:", dict(Counter(r["category"] for r in rows)))
    for k in sorted(stats):
        if k.startswith("skip"):
            print(f"  {k}: {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
