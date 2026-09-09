"""NuInstruct blank-view 진단. 근거뷰를 가릴 때와 대조뷰를 가릴 때의 성능 낙폭 차이.

  drop_ev   = full - mask_evidence   근거뷰를 가렸을 때 떨어지는 폭
  drop_ctrl = full - mask_control    무관한 뷰를 가렸을 때 떨어지는 폭
  gap       = drop_ev - drop_ctrl    이게 클수록 답이 실제로 그 뷰를 보고 나온 것

gap 이 0 이면 모델이 뷰를 구분하지 않는다는 뜻이다 — 텍스트 사전확률이나 다른
뷰의 문맥으로 답을 만들고 있을 수 있다.

MAE 는 낮을수록 좋으므로 부호를 뒤집어 "성능"으로 통일한 뒤 낙폭을 계산한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.nuinstruct_eval import (ACC_TASKS, BLEU_TASKS, MAE_TASKS, acc_score,
                                   ap_at_iou, mae_score, parse_objs)


def score_rows(preds, meta):
    """task 별 점수. MAE 는 성능 방향(음수)으로 뒤집어 반환한다."""
    per = defaultdict(list)
    bleu = defaultdict(list)
    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        task, gt, ans = m["task"], m["solution"], p["llm-response"]
        if task in ACC_TASKS:
            per[task].append(1.0 if acc_score(ans, gt, task) else 0.0)
        elif task in MAE_TASKS:
            v = mae_score(ans, gt, task)
            per[task].append(0.0 if (v is None or isinstance(v, tuple)) else -v)
        elif task in BLEU_TASKS:
            bleu[task].append((ans, gt))
        else:
            per[task].append(ap_at_iou(parse_objs(ans), parse_objs(gt), 0.5))
    out = {t: sum(v) / len(v) for t, v in per.items() if v}
    if bleu:
        from pycocoevalcap.bleu.bleu import Bleu
        for t, items in bleu.items():
            r = {str(i): [g] for i, (_, g) in enumerate(items)}
            c = {str(i): [a] for i, (a, _) in enumerate(items)}
            b, _ = Bleu(4).compute_score(r, c)
            out[t] = b[3]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="bv_<tag> 접두사")
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/nuins_eval_vg.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = {r["id"]: r for r in json.load(open(args.dataset))}
    cond = {}
    for c in ["none", "evidence", "control"]:
        p = f"{args.prefix}_{c}.json"
        if not os.path.exists(p):
            print(f"없음: {p}")
            return 1
        cond[c] = score_rows(json.load(open(p)), meta)

    tasks = sorted(set().union(*[set(v) for v in cond.values()]))
    res = {"label": args.label, "per_task": {}}
    print(f"\n=== {args.label or args.prefix} ===")
    print(f"  {'task':28s}{'full':>9s}{'mask_ev':>9s}{'mask_ctrl':>10s}"
          f"{'drop_ev':>9s}{'drop_ctrl':>10s}{'gap':>9s}")
    for t in tasks:
        f = cond["none"].get(t)
        e = cond["evidence"].get(t)
        c = cond["control"].get(t)
        if None in (f, e, c):
            continue
        de, dc = f - e, f - c
        res["per_task"][t] = {"full": round(f, 4), "mask_ev": round(e, 4),
                              "mask_ctrl": round(c, 4), "drop_ev": round(de, 4),
                              "drop_ctrl": round(dc, 4), "gap": round(de - dc, 4)}
        print(f"  {t:28s}{f:9.4f}{e:9.4f}{c:10.4f}{de:+9.4f}{dc:+10.4f}{de-dc:+9.4f}")

    # 전체 평균 (task 별 점수의 단순 평균 — 지표 축이 섞여 있으므로 상대 비교용)
    if res["per_task"]:
        n = len(res["per_task"])
        for k in ["full", "mask_ev", "mask_ctrl", "drop_ev", "drop_ctrl", "gap"]:
            res[k] = round(sum(d[k] for d in res["per_task"].values()) / n, 4)
        print(f"  {'평균':28s}{res['full']:9.4f}{res['mask_ev']:9.4f}"
              f"{res['mask_ctrl']:10.4f}{res['drop_ev']:+9.4f}"
              f"{res['drop_ctrl']:+10.4f}{res['gap']:+9.4f}")

    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
