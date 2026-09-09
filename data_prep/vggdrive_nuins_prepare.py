"""VGGDrive 가 배포한 NuInstruct test JSON 을 우리 추론 형식으로 바꾼다.

VGGDrive(HF: wang-jie825/VGGDrive_Qwen_json/nuScenes_cache)의 전처리는 원본
NuInstruct 와 여러 곳이 다르다. 그들 Table 2 와 직접 비교하려면 그 차이를 그대로
보존해야 한다:
  - 뷰 순서가 F, FL, FR, B, BL, BR 이다 (원본은 FL, F, FR, BL, B, BR)
  - 정답이 카메라를 이름으로 쓴다: <car>[CAM_FRONT_LEFT,688,450,896,540]
    (원본은 인덱스 <car>[c0,...])
  - 프롬프트에 뷰 순서를 설명하는 문장이 붙어 있고 시스템 프롬프트가 없다
  - test 가 지표별 4개 파일로 나뉘어 있다 (accuracy/mae/map/reasoning)
  - 이미지에 resized_height/width 448 이 지정돼 있다
따라서 시스템 프롬프트를 새로 붙이지 않고 그들 user 메시지를 그대로 쓴다.

근거뷰 라벨은 정답의 카메라 *이름* 에서 뽑고, 그 파일의 뷰 순서로 인덱싱한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

# VGGDrive 프롬프트가 명시하는 순서
VG_VIEW_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
NAME_RE = re.compile(r"\[\s*(CAM_[A-Z_]+)\s*,")
# 지표별 파일 -> 우리 채점기의 task 축
FILE_METRIC = {"accuracytask": "Accuracy", "maetask": "MAE",
               "maptask": "MAP", "reasoningtask": "BLEU"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--nusc_root", default="/nuscenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check_images", action="store_true")
    args = ap.parse_args()

    rows, stats = [], Counter()
    for tag, metric in FILE_METRIC.items():
        p = os.path.join(args.src_dir, f"NuInstruct_Qwen_test_1frames_{tag}.json")
        if not os.path.exists(p):
            stats[f"skip/no_file_{tag}"] += 1
            continue
        for r in json.load(open(p)):
            msgs = r["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            asst = next((m for m in msgs if m["role"] == "assistant"), None)
            gold = asst["content"] if asst else ""
            if isinstance(gold, list):
                gold = " ".join(c.get("text", "") for c in gold)

            images, texts = [], []
            for c in user["content"]:
                if c.get("type") == "image":
                    rel = c["image"].replace("file://", "")
                    rel = re.sub(r"^.*?samples/", "samples/", rel)
                    images.append(os.path.join(args.nusc_root, rel))
                elif c.get("type") == "text":
                    texts.append(c["text"])
            if len(images) != 6:
                stats[f"skip/n_images_{len(images)}"] += 1
                continue
            if args.check_images and not all(os.path.exists(x) for x in images):
                stats["skip/image_absent"] += 1
                continue
            question = " ".join(texts).strip()

            # 근거뷰: 정답의 카메라 이름을 이 파일의 뷰 순서로 인덱싱
            ev = sorted({VG_VIEW_ORDER.index(n) for n in NAME_RE.findall(gold or "")
                         if n in VG_VIEW_ORDER})
            rows.append({
                # 4개 파일이 각자 00000000 부터 번호를 다시 매겨서 id 가 겹친다
                # (16,129건 중 9,486건 중복). 파일 태그를 붙여 고유하게 만든다.
                "id": f"{tag}::{r['id']}",
                "orig_id": r["id"],
                "sample_token": r.get("sample_token"),
                "vg_metric": metric,          # 채점기가 이 축으로 나눈다
                "task": f"vg-{tag}",
                "category": tag.replace("task", ""),
                "question": question,
                "solution": gold,
                "images": images,
                # VGGDrive 는 시스템 프롬프트를 쓰지 않는다. 그대로 재현한다.
                "prompt": [{"role": "user",
                            "content": [{"type": "image"} for _ in images]
                            + [{"type": "text", "text": question}]}],
                "evidence_views": ev,
                "vg_usable": 0 < len(ev) < 6,
            })
            stats[f"kept/{tag}"] += 1

    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"총 {len(rows)}행 -> {args.out}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    u = sum(1 for r in rows if r["vg_usable"])
    print(f"vg_usable: {u} ({100*u/max(len(rows),1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
