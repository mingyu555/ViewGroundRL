"""DriveLM 4개 카테고리 SFT 학습셋. 평가 GT 와 같은 원문 답변을 타깃으로 쓴다.

기존 dlm_sft_train.json 은 교사 모델이 만든 <think> CoT 를 타깃으로 삼는데, 평가
GT 는 "No." / "Stationary." 같은 짧은 원문이라 형식이 어긋난다. 여기서는 원문
답변을 그대로 타깃으로 둔다.

평가에 쓰는 604개 held-out 프레임은 제외한다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.drivelm_prepare import iter_qa, resolve_paths  # noqa: E402
from vgrl.prompt_format import SYSTEM_DRIVELM  # noqa: E402
from vgrl.views import CAMERAS, evidence_views  # noqa: E402

CATS = ["perception", "prediction", "planning", "behavior"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivelm_json", required=True)
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--exclude_frames_from", required=True)
    # 프레임 목록만으로 거르면 그 목록이 val 씬 전체를 덮지 못한다. 실측: val 씬 2개가
    # 학습에 새어들어 학습 3행이 평가 씬 소속이었다 (프레임은 겹치지 않았다).
    # scene_split.json 의 val_scenes 로 씬 단위까지 잘라야 완전 분리가 된다.
    ap.add_argument("--scene_split", default=None,
                    help="scene_split.json — val_scenes 소속 프레임을 전부 제외")
    ap.add_argument("--per_category", type=int, default=3500)
    ap.add_argument("--val_fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val_frames = {r["id"].split("::")[0]
                  for r in json.load(open(args.exclude_frames_from))}
    if args.scene_split:
        sp = json.load(open(args.scene_split))
        val_scenes = set(sp["val_scenes"])
        added = {fr for fr, sc in sp["frame2scene"].items() if sc in val_scenes}
        print(f"씬 분할로 추가 제외 {len(added - val_frames)}개 프레임")
        val_frames |= added
    print(f"제외할 held-out 프레임 {len(val_frames)}개")

    data = json.load(open(args.drivelm_json))
    by = defaultdict(list)
    stats = Counter()
    for rec in iter_qa(data):
        if rec["frame_token"] in val_frames:
            stats["skip/heldout"] += 1
            continue
        images = resolve_paths(rec["image_paths"], args.nusc_root)
        if images is None:
            stats["skip/no_image"] += 1
            continue
        ev = evidence_views(rec["question"], rec["answer"], source="both")
        by[rec["category"]].append({
            "id": f"{rec['frame_token']}::{rec['category']}::{rec['qa_index']}",
            "images": images,
            "system": SYSTEM_DRIVELM,
            "question": rec["question"],
            "solution": rec["answer"],
            # 평가 프롬프트와 정확히 같은 구성: system(카메라 순서 설명) + 6뷰 + 질문.
            # train_sft.to_conversation 이 <image> 자리표시자를 이미지 파트로 바꾼다.
            "conversations": [
                {"from": "human", "value": "<image>" * len(CAMERAS) + rec["question"]},
                {"from": "gpt", "value": rec["answer"]},
            ],
            "category": rec["category"],
            "evidence_views": ev,
            "vg_usable": 0 < len(ev) < len(CAMERAS),
        })

    rng = random.Random(args.seed)
    rows = []
    print("\n카테고리별:")
    for c in CATS:
        v = by[c]
        rng.shuffle(v)
        take = v[: args.per_category]
        rows += take
        print(f"  {c:12s} 가용 {len(v):7d}  →  {len(take):5d}")

    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val, train = rows[:n_val], rows[n_val:]
    base, ext = os.path.splitext(args.out)
    json.dump(train, open(args.out, "w"), ensure_ascii=False)
    json.dump(val, open(base + "_val" + ext, "w"), ensure_ascii=False)
    print(f"\ntrain {len(train)} -> {args.out}")
    print(f"val   {len(val)} -> {base}_val{ext}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
