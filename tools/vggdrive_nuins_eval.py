"""VGGDrive 가 배포한 NuInstruct test 셋 채점. 그들 Table 2 와 축을 맞춘다.

원본 NuInstruct 채점기(tools/nuinstruct_eval.py)와 다른 점:
  - 정답이 카메라를 이름으로 쓴다: <car>[CAM_FRONT_LEFT,688,450,896,540]
  - test 가 지표별로 이미 나뉘어 있어 task 이름으로 지표를 고를 필요가 없다
    (vg_metric 컬럼을 그대로 쓴다)
  - Accuracy 그룹에 'closest'(객체 지목)와 'status'(문자열) 두 종류가 섞여 있다
  - MAE 그룹에 'How many'(개수)와 'distance between'(2D 벡터)이 섞여 있다

VGGDrive Table 2 (Qwen2.5-VL-7B 계열):
  Baseline   MAE 4.35 / Acc 47.71 / MAP  6.15 / BLEU 75.75 / Avg* 31.32
  VGGT-Dist  MAE 3.73 / Acc 56.21 / MAP 28.51 / BLEU 79.23 / Avg* 40.06
  VGGDrive   MAE 3.08 / Acc 56.37 / MAP 37.49 / BLEU 81.13 / Avg* 42.98
  Average* = max((Acc + MAP + BLEU - MAE) / 4, 0)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.nuinstruct_eval import ap_at_iou, iou  # noqa: E402

# <class>[CAM_NAME,x1,y1,x2,y2]
OBJ_RE = re.compile(r"<\s*([A-Za-z_]+)\s*>\s*\[\s*(CAM_[A-Z_]+)\s*,\s*"
                    r"([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
VEC_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
WS = re.compile(r"\s+")


def parse_objs(text: str):
    out = []
    for cls, cam, a, b, c, d in OBJ_RE.findall(text or ""):
        try:
            out.append((cls.lower(), cam, (float(a), float(b), float(c), float(d))))
        except ValueError:
            continue
    return out


def norm(s: str) -> str:
    return WS.sub(" ", re.sub(r"[.\s]+$", "", (s or "").strip().lower()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/vgnuins_eval.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocoevalcap.bleu.bleu import Bleu

    meta = {r["id"]: r for r in json.load(open(args.dataset))}
    preds = json.load(open(args.pred))

    acc, mae, mp = [], [], []
    bleu_items = []
    sub = defaultdict(lambda: {"n": 0, "vals": []})
    unparsed = defaultdict(int)

    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        gold, ans, met, q = m["solution"], p["llm-response"], m["vg_metric"], m["question"]

        if met == "Accuracy":
            if "closest" in q:
                go, po = parse_objs(gold), parse_objs(ans)
                # 가장 가까운 객체 하나: 클래스 + 카메라 일치
                ok = bool(go) and bool(po) and po[0][0] == go[0][0] and po[0][1] == go[0][1]
                key = "closest"
            else:
                ok = norm(ans) == norm(gold)
                key = "status"
            acc.append(1.0 if ok else 0.0)
            sub[key]["n"] += 1; sub[key]["vals"].append(1.0 if ok else 0.0)

        elif met == "MAE":
            if "distance between" in q:
                pg, gg = VEC_RE.search(ans), VEC_RE.search(gold)
                key = "distance"
                if gg is None:
                    continue
                if pg is None:
                    unparsed[key] += 1; continue
                v = (abs(float(pg.group(1)) - float(gg.group(1)))
                     + abs(float(pg.group(2)) - float(gg.group(2)))) / 2.0
            else:
                key = "count"
                pn = NUM_RE.search((ans or "").replace(",", ""))
                gn = NUM_RE.search((gold or "").replace(",", ""))
                if gn is None:
                    continue
                if pn is None:
                    unparsed[key] += 1; continue
                v = abs(float(pn.group(0)) - float(gn.group(0)))
            mae.append(v)
            sub[key]["n"] += 1; sub[key]["vals"].append(v)

        elif met == "MAP":
            v = ap_at_iou(parse_objs(ans), parse_objs(gold), args.iou)
            mp.append(v)
            sub["risk"]["n"] += 1; sub["risk"]["vals"].append(v)

        else:                                     # BLEU
            bleu_items.append((ans, gold))

    res = {"label": args.label, "pred_file": args.pred, "iou": args.iou,
           "n_pred": len(preds), "unparsed": dict(unparsed)}
    res["MAE"] = round(sum(mae) / len(mae), 2) if mae else None
    res["Accuracy"] = round(100 * sum(acc) / len(acc), 2) if acc else None
    res["MAP"] = round(100 * sum(mp) / len(mp), 2) if mp else None
    if bleu_items:
        r = {str(i): [g] for i, (_, g) in enumerate(bleu_items)}
        c = {str(i): [a] for i, (a, _) in enumerate(bleu_items)}
        b, _ = Bleu(4).compute_score(r, c)
        res["BLEU"] = round(100 * b[3], 2)
        res["BLEU-1..4"] = [round(100 * x, 2) for x in b]
    if None not in (res["MAE"], res["Accuracy"], res["MAP"], res.get("BLEU")):
        res["Average*"] = round(max((res["Accuracy"] + res["MAP"] + res["BLEU"]
                                     - res["MAE"]) / 4.0, 0.0), 2)
    res["breakdown"] = {k: {"n": v["n"],
                            "mean": round(sum(v["vals"]) / len(v["vals"]), 4) if v["vals"] else None}
                        for k, v in sub.items()}

    print(f"\n=== {args.label or args.pred} ===")
    print(f"  MAE↓ {res['MAE']}   Accuracy↑ {res['Accuracy']}   MAP↑ {res['MAP']}   "
          f"BLEU↑ {res.get('BLEU')}   Average*↑ {res.get('Average*')}")
    print("\n  세부:")
    for k, v in res["breakdown"].items():
        print(f"    {k:10s} n={v['n']:5d}  mean={v['mean']}")
    if unparsed:
        print("  미파싱:", dict(unparsed))
    print("\n  VGGDrive Table 2 (동일 베이스 Qwen2.5-VL-7B):")
    print("    Baseline  MAE 4.35 / Acc 47.71 / MAP  6.15 / BLEU 75.75 / Avg* 31.32")
    print("    VGGDrive  MAE 3.08 / Acc 56.37 / MAP 37.49 / BLEU 81.13 / Avg* 42.98")

    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
