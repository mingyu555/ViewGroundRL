"""DriveLM 4개 카테고리 평가셋을 만든다 (perception/prediction/planning/behavior).

기존 dlm_scene_val.json 은 3개 카테고리(각 1200행)만 있고 vg_usable=True 로 걸러져
있다. behavior 는 질문이 카메라를 지목하지 않아 근거뷰가 0개라 그 필터에서 통째로
빠진다 — 확인 결과 606/606 이 근거뷰 0개다.

여기서는 같은 held-out 프레임(604개)을 그대로 쓰되 필터 없이 4개 카테고리를 전부
담는다. blank-view 진단용이 아니라 일반 성능 평가용이므로 vg_usable 은 표시만 하고
거르지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.drivelm_prepare import iter_qa, resolve_paths  # noqa: E402
from vgrl.prompt_format import SYSTEM_DRIVELM, VIEW_HEADER  # noqa: E402
from vgrl.views import CAMERAS, evidence_views  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivelm_json", required=True)
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--val_frames_from", required=True,
                    help="held-out 프레임 토큰을 가져올 기존 val json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = {r["id"].split("::")[0] for r in json.load(open(args.val_frames_from))}
    print(f"held-out 프레임 {len(frames)}개")

    data = json.load(open(args.drivelm_json))
    rows, stats = [], Counter()
    for rec in iter_qa(data):
        if rec["frame_token"] not in frames:
            continue
        stats[f"qa/{rec['category']}"] += 1
        images = resolve_paths(rec["image_paths"], args.nusc_root)
        if images is None:
            stats["skip/no_image"] += 1
            continue
        ev = evidence_views(rec["question"], rec["answer"], source="both")
        rows.append({
            "id": f"{rec['frame_token']}::{rec['category']}::{rec['qa_index']}",
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_DRIVELM}]},
                {"role": "user", "content": [{"type": "image"} for _ in CAMERAS]
                 + [{"type": "text", "text": rec["question"]}]},
            ],
            "question": rec["question"],
            "images": images,
            "solution": rec["answer"],
            "category": rec["category"],
            "evidence_views": ev,
            "vg_usable": 0 < len(ev) < len(CAMERAS),
        })
        stats[f"kept/{rec['category']}"] += 1

    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"총 {len(rows)}행 -> {args.out}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    usable = Counter((r["category"], r["vg_usable"]) for r in rows)
    print("\n카테고리별 vg_usable:")
    for c in ["perception", "prediction", "planning", "behavior"]:
        t, f = usable[(c, True)], usable[(c, False)]
        print(f"  {c:12s} True {t:5d}  False {f:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
