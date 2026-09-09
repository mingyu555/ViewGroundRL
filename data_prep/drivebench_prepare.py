"""DriveBench(drive-bench/arena)를 우리 추론 스크립트가 먹는 형식으로 바꾼다.

DriveBench 는 평가 전용이고 공식 DriveLM `tag` 를 갖고 있어서 공식 채점 경로를
그대로 쓸 수 있다 (tag0=accuracy, tag1=ChatGPT, tag2=language, tag3=match).

우리 학습 데이터와 형식이 다른 점을 그대로 보존한다 — 고쳐 맞추면 벤치마크를
왜곡하게 된다:
  - 좌표가 정규화(0~1)다. DriveLM train 은 픽셀(0~1600, 0~900)이다.
  - 문항 1,461건 중 400건이 객관식이고 정답이 "A"/"C" 같은 한 글자다.
  - tag3(match) 정답에 좌표 태그가 아예 없다 → 공식 match 는 계산 불가.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.views import CAMERAS  # noqa: E402


# DriveBench 공식 추론 시스템 프롬프트 (논문 Figure 20 축자).
# 우리 기본 SYSTEM_DRIVELM 을 쓰면 안 된다 — 이 프롬프트가 (a) 좌표가 정규화라는
# 것과 (b) 객관식/is-질문에 "설명을 붙이라"는 것을 지시하는데, GPT 채점 루브릭
# 100점 중 50점이 그 설명에 걸려 있다. 설명 없는 답은 2~5번 항목이 0점 처리된다.
DRIVEBENCH_SYSTEM = (
    "You are a smart autonomous driving assistant responsible for analyzing and "
    "responding to driving scenarios. You are provided with up to six camera images "
    "in the sequence [CAM FRONT, CAM FRONT LEFT, CAM FRONT RIGHT, CAM BACK, "
    "CAM BACK LEFT, CAM BACK RIGHT]. Each image has normalized coordinates from "
    "[0, 1], with (0,0) at the top left and (1,1) at the bottom right.\n"
    "Instructions:\n"
    "1. Answer Requirements:\n"
    "- For multiple-choice questions, provide the selected answer choice along with "
    "an explanation.\n"
    "- For \"is\" or \"is not\" questions, respond with a \"Yes\" or \"No\", along with "
    "an explanation.\n"
    "- For open-ended perception and prediction questions, related objects to which "
    "the camera.\n"
    "2. Key Information for Driving Context:\n"
    "- When answering, focus on object attributes (e.g., categories, statuses, visual "
    "descriptions) and motions (e.g., speed, action, acceleration) relevant to driving "
    "decision-making\n"
    "Use the images and coordinate information to respond accurately to questions "
    "related to perception, prediction, planning, or behavior, based on the question "
    "requirements."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivebench_json", required=True)
    ap.add_argument("--nusc_root", default="/nuscenes",
                    help="컨테이너 안의 nuScenes 루트 (data/nuscenes/ 를 이걸로 치환)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.load(open(args.drivebench_json))
    rows, stats = [], Counter()
    for i, r in enumerate(src):
        # image_path 는 카메라 이름 -> 상대경로 dict. CAMERAS 순서로 정렬한다.
        try:
            images = [os.path.join(args.nusc_root,
                                   r["image_path"][c].replace("data/nuscenes/", ""))
                      for c in CAMERAS]
        except KeyError as e:
            stats[f"skip/missing_cam_{e}"] += 1
            continue
        rows.append({
            "id": f"{r['frame_token']}::{r['question_type']}::{i}",
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": DRIVEBENCH_SYSTEM}]},
                {"role": "user", "content": [{"type": "image"} for _ in CAMERAS]
                 + [{"type": "text", "text": r["question"]}]},
            ],
            "question": r["question"],
            "images": images,
            "solution": r["answer"],
            "category": r["question_type"],
            "tag": r["tag"],
            "scene_token": r["scene_token"],
            "frame_token": r["frame_token"],
        })
        stats[f"kept/{r['question_type']}"] += 1

    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"{len(rows)}행 -> {args.out}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print("\ntag 분포:", dict(Counter(tuple(r["tag"]) for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
