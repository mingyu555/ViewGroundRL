"""DriveLM 을 씬 단위로 나눈다. 이후 모든 단계가 이 split 을 공유한다.

왜 씬 단위인가: 앞선 실험은 `drivelm_prepare.py` 가 QA 행을 셔플해 2% 를 떼어냈고,
프레임당 QA 가 평균 12개라 같은 프레임의 다른 질문이 train 과 val 에 흩어졌다.
실측하니 perception val 918 프레임이 train 과 100% 겹쳤고, val 질문 1,038개 중
199개는 train 에 같은 문자열로 존재했다. 모델이 RL 중 그 프레임의 6뷰를 이미 본
상태에서 낸 수치였다.

프레임 단위로 나눠도 부족하다 — 한 씬의 key frame 들은 몇 초 간격이라 배경·차량이
거의 같다. 그래서 씬 토큰으로 자른다.

    python data_prep/scene_split.py --drivelm_json <raw> --out_json <split.json>
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivelm_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--val_scene_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = json.load(open(args.drivelm_json))
    scenes = sorted(raw)                      # 정렬 후 시드 셔플 -> 재현 가능
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_val = int(len(scenes) * args.val_scene_fraction)
    val_scenes, train_scenes = set(scenes[:n_val]), set(scenes[n_val:])

    frame2scene, counts = {}, Counter()
    for sc, v in raw.items():
        for fr, fd in (v.get("key_frames") or {}).items():
            frame2scene[fr] = sc
            n = sum(len(x) for x in (fd.get("QA") or {}).values())
            counts["val" if sc in val_scenes else "train"] += n

    out = {
        "seed": args.seed,
        "val_scene_fraction": args.val_scene_fraction,
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "frame2scene": frame2scene,
    }
    json.dump(out, open(args.out_json, "w"))

    tf = sum(1 for f, s in frame2scene.items() if s in train_scenes)
    vf = len(frame2scene) - tf
    print(f"씬   train {len(train_scenes)} / val {len(val_scenes)}")
    print(f"프레임 train {tf} / val {vf}")
    print(f"QA   train {counts['train']} / val {counts['val']}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
