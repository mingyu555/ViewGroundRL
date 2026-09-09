"""NuInstruct 공식 지표. 전부 규칙 기반이라 GPT 가 필요 없다.

논문(arXiv:2401.00988) Table 2 의 배정 그대로:
  Perception  Distance, Speeds, Instance Number   MAE↓
              Closest, Status, Same Road          Accuracy↑
  Prediction  Motion Ego, Motion Others           MAE↓
              Status Ego, Status Others           Accuracy↑
  Risk        All                                 MAP↑
  Reasoning   All                                 BLEU↑

VGGDrive(arXiv:2602.20794) Table 2 가 종합 점수를 이렇게 정의한다:
  Average* = max((Accuracy + MAP + BLEU - MAE) / 4, 0)
같은 논문의 Qwen2.5-VL-7B SFT "Baseline" 행이 우리 대조군이다:
  MAE 4.35 / Accuracy 47.71 / MAP 6.15 / BLEU 75.75 / Average* 31.32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# <class>[cN,x1,y1,x2,y2]  (좌표는 task 에 따라 픽셀 또는 정규화)
OBJ_RE = re.compile(r"<\s*([A-Za-z_]+)\s*>\s*\[\s*c(\d+)\s*,\s*"
                    r"([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
VEC_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

MAE_TASKS = {"perception-distance", "perception-speed", "perception-instance_count",
             "prediction-motion_ego", "prediction-motion_other"}
ACC_TASKS = {"perception-closest", "perception-status", "perception-in_the_same_road",
             "prediction-status_ego", "prediction-status_others"}
BLEU_TASKS = {"reasoning-reasoning"}
# risk-* 는 전부 MAP


def parse_objs(text: str):
    """<class>[cN,x1,y1,x2,y2] 목록을 (class, cam, box) 로."""
    out = []
    for cls, cam, a, b, c, d in OBJ_RE.findall(text or ""):
        try:
            out.append((cls.lower(), int(cam), (float(a), float(b), float(c), float(d))))
        except ValueError:
            continue
    return out


def parse_vec(text: str):
    m = VEC_RE.search(text or "")
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def parse_num(text: str):
    m = NUM_RE.search((text or "").replace(",", ""))
    return float(m.group(0)) if m else None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + \
         max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def mae_score(pred: str, gt: str, task: str):
    """MAE. 벡터형(distance/motion)은 L1 노름 차, 스칼라형은 절대차."""
    if task in ("perception-distance", "prediction-motion_ego", "prediction-motion_other"):
        p, g = parse_vec(pred), parse_vec(gt)
        if g is None:
            return None
        if p is None:
            return None, True          # 파싱 실패 표시
        return (abs(p[0] - g[0]) + abs(p[1] - g[1])) / 2.0
    p, g = parse_num(pred), parse_num(gt)
    if g is None:
        return None
    if p is None:
        return None, True
    return abs(p - g)


def acc_score(pred: str, gt: str, task: str) -> bool:
    """Accuracy. closest 는 클래스+카메라 일치, 나머지는 정규화 문자열 일치."""
    if task == "perception-closest":
        po, go = parse_objs(pred), parse_objs(gt)
        if not go:
            return False
        if not po:
            return False
        # 가장 가까운 객체 하나를 묻는다: 클래스와 카메라가 맞으면 정답
        return po[0][0] == go[0][0] and po[0][1] == go[0][1]
    if task == "perception-in_the_same_road":
        # 이 task 는 두 종류가 섞여 있다:
        #   "Does any objects ...?"  -> 정답이 yes / no
        #   "Which objects are ...?" -> 정답이 <class>[cN,...] 목록
        gn = re.sub(r"[.\s]+$", "", (gt or "").strip().lower())
        pn = re.sub(r"[.\s]+$", "", (pred or "").strip().lower())
        if gn in ("yes", "no", "none"):
            # 모델이 "yes, there are ..." 처럼 답할 수 있으므로 앞머리로 판정
            want = "yes" if gn == "yes" else "no"
            first = re.match(r"^(yes|no)\b", pn)
            if first:
                return first.group(1) == want
            # yes/no 를 안 쓰면 객체를 나열했는지로 대신 본다
            has = bool(parse_objs(pred))
            return has if want == "yes" else not has
        gset = {c for _, c, _ in parse_objs(gt)}
        pset = {c for _, c, _ in parse_objs(pred)}
        return bool(gset) and gset == pset
    a = re.sub(r"[.\s]+$", "", (pred or "").strip().lower())
    b = re.sub(r"[.\s]+$", "", (gt or "").strip().lower())
    return a == b


def ap_at_iou(preds, gts, thr=0.5) -> float:
    """한 문항의 AP. 예측 순서를 신뢰도로 보고 greedy 매칭 후 PR 면적 근사."""
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    used = [False] * len(gts)
    tps = []
    for cls, cam, box in preds:
        best, bi = 0.0, -1
        for i, (gc, gcam, gbox) in enumerate(gts):
            if used[i] or gc != cls or gcam != cam:
                continue
            v = iou(box, gbox)
            if v > best:
                best, bi = v, i
        if bi >= 0 and best >= thr:
            used[bi] = True
            tps.append(1)
        else:
            tps.append(0)
    # 11-point 보간 없이 면적 근사
    cum_tp = 0
    prec, rec = [], []
    for i, t in enumerate(tps):
        cum_tp += t
        prec.append(cum_tp / (i + 1))
        rec.append(cum_tp / len(gts))
    ap, prev_r = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/nuins_eval.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocoevalcap.bleu.bleu import Bleu

    meta = {r["id"]: r for r in json.load(open(args.dataset))}
    preds = json.load(open(args.pred))

    per_task = defaultdict(lambda: {"n": 0, "vals": [], "unparsed": 0})
    bleu_items = []
    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        task, gt = m["task"], m["solution"]
        ans = p["llm-response"]
        d = per_task[task]
        d["n"] += 1
        if task in MAE_TASKS:
            v = mae_score(ans, gt, task)
            if isinstance(v, tuple):
                d["unparsed"] += 1
            elif v is not None:
                d["vals"].append(v)
        elif task in ACC_TASKS:
            d["vals"].append(1.0 if acc_score(ans, gt, task) else 0.0)
        elif task in BLEU_TASKS:
            bleu_items.append((ans, gt))
        else:                                    # risk-*
            d["vals"].append(ap_at_iou(parse_objs(ans), parse_objs(gt), args.iou))

    res = {"label": args.label, "pred_file": args.pred, "iou": args.iou, "per_task": {}}
    for t, d in per_task.items():
        kind = ("MAE" if t in MAE_TASKS else "Accuracy" if t in ACC_TASKS
                else "BLEU" if t in BLEU_TASKS else "MAP")
        v = d["vals"]
        score = (sum(v) / len(v)) if v else None
        if kind == "Accuracy" or kind == "MAP":
            score = score * 100 if score is not None else None
        res["per_task"][t] = {"metric": kind, "n": d["n"], "scored": len(v),
                              "unparsed": d["unparsed"],
                              "score": round(score, 4) if score is not None else None}
    if bleu_items:
        r = {str(i): [g] for i, (_, g) in enumerate(bleu_items)}
        c = {str(i): [a] for i, (a, _) in enumerate(bleu_items)}
        b, _ = Bleu(4).compute_score(r, c)
        res["per_task"]["reasoning-reasoning"] = {
            "metric": "BLEU", "n": len(bleu_items), "scored": len(bleu_items),
            "unparsed": 0, "score": round(b[3] * 100, 4),
            "BLEU-1..4": [round(x * 100, 2) for x in b]}

    # 지표군별 집계 (논문 Table 2 축)
    def agg(kind):
        v = [d["score"] for d in res["per_task"].values()
             if d["metric"] == kind and d["score"] is not None]
        return round(sum(v) / len(v), 2) if v else None

    mae, acc, mp, bl = agg("MAE"), agg("Accuracy"), agg("MAP"), agg("BLEU")
    res["summary"] = {"MAE": mae, "Accuracy": acc, "MAP": mp, "BLEU": bl}
    if None not in (mae, acc, mp, bl):
        res["summary"]["Average*"] = round(max((acc + mp + bl - mae) / 4.0, 0.0), 2)

    print(f"\n=== {args.label or args.pred} ===")
    print(f"  {'task':30s}{'metric':>10s}{'n':>7s}{'score':>10s}{'미파싱':>8s}")
    for t in sorted(res["per_task"]):
        d = res["per_task"][t]
        print(f"  {t:30s}{d['metric']:>10s}{d['n']:7d}"
              f"{(d['score'] if d['score'] is not None else '—'):>10}{d['unparsed']:8d}")
    s = res["summary"]
    print(f"\n  MAE↓ {s['MAE']}   Accuracy↑ {s['Accuracy']}   MAP↑ {s['MAP']}   "
          f"BLEU↑ {s['BLEU']}   Average*↑ {s.get('Average*')}")
    print("  (VGGDrive Table 2 Baseline: MAE 4.35 / Acc 47.71 / MAP 6.15 / "
          "BLEU 75.75 / Avg* 31.32)")

    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
