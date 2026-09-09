"""DriveBench 공식 채점. challenge/evaluation.py 의 tag 라우팅을 그대로 쓴다.

    tag 0 → accuracy   eval_acc(): answer == GT 완전 일치
    tag 1 → ChatGPT    GPT API 필요 → 계산 불가
    tag 2 → language   BLEU-1~4 / ROUGE-L / CIDEr
    tag 3 → match      match_result 좌표 F1

최종 점수 = 0.4*chatgpt + 0.2*language + 0.2*match + 0.2*accuracy
  language = mean(BLEU1~4)/3 + ROUGE_L/3 + CIDEr/10/3
  match    = (좌표F1*100 + ChatGPT)/2 /100

DriveBench 특성상 미리 알아야 할 것:
  - tag3 정답에 좌표 태그가 없다 → match 는 대상이 0 이라 계산 불가.
  - tag0 600건 중 400건이 객관식이고 정답이 "A"/"C" 한 글자다.
  - tag0 비객관식 200건은 194건이 "No." 다 (97%) — 불균형이 극심하다.
  - 좌표가 정규화(0~1)라 픽셀로 학습한 모델과 축이 다르다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.official_metrics import match_result  # noqa: E402
from vgrl.prompt_format import extract_answer  # noqa: E402

_PUNCT = re.compile(r"([.,!?;:()\[\]{}\"])")
_WS = re.compile(r"\s+")
_LETTER = re.compile(r"\b([A-D])\b")


def ptb_like(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(r" \1 ", (s or "").lower())).strip()


def norm(s: str) -> str:
    return _WS.sub(" ", re.sub(r"[.,!?;:\s]+$", "", (s or "").strip().lower()))


def mcq_letter(s: str) -> str | None:
    """객관식 답에서 선택지 문자를 뽑는다. 모델이 'A. Going ahead.' 처럼 답할 수 있다."""
    m = _LETTER.search((s or "").strip().upper())
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/drivebench_test.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    meta = {r["id"]: r for r in json.load(open(args.dataset))}
    preds = json.load(open(args.pred))

    routed = defaultdict(list)          # tag -> [(pred_answer, gt, is_mcq)]
    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        a = extract_answer(p["llm-response"]) or p["llm-response"]
        is_mcq = "select the correct answer" in m["question"]
        for t in m["tag"]:
            routed[t].append((a, m["solution"], is_mcq))

    out = {"label": args.label, "pred_file": args.pred, "n_pred": len(preds)}

    # ---- tag 0: accuracy (공식은 완전 일치) ----
    items = routed.get(0, [])
    if items:
        exact = sum((a or "").strip() == (g or "").strip() for a, g, _ in items)
        normed = sum(norm(a) == norm(g) for a, g, _ in items)
        mcq = [(a, g) for a, g, k in items if k]
        nonmcq = [(a, g) for a, g, k in items if not k]
        mcq_hit = sum(mcq_letter(a) == norm(g).upper() for a, g in mcq)
        out["accuracy"] = {
            "n": len(items),
            "official_exact": round(exact / len(items), 4),
            "normalized": round(normed / len(items), 4),
            "mcq_n": len(mcq),
            "mcq_letter_match": round(mcq_hit / len(mcq), 4) if mcq else None,
            "nonmcq_n": len(nonmcq),
            "nonmcq_normalized": round(
                sum(norm(a) == norm(g) for a, g in nonmcq) / len(nonmcq), 4) if nonmcq else None,
        }

    # ---- tag 2: language ----
    items = routed.get(2, [])
    if items:
        res = {str(i): [ptb_like(a)] for i, (a, _, _) in enumerate(items)}
        ref = {str(i): [ptb_like(g)] for i, (_, g, _) in enumerate(items)}
        bleu, _ = Bleu(4).compute_score(ref, res)
        rouge, _ = Rouge().compute_score(ref, res)
        cider, _ = Cider().compute_score(ref, res)
        out["language"] = {
            "n": len(items),
            "BLEU-1": round(bleu[0], 4), "BLEU-2": round(bleu[1], 4),
            "BLEU-3": round(bleu[2], 4), "BLEU-4": round(bleu[3], 4),
            "ROUGE-L": round(rouge, 4), "CIDEr": round(cider, 4),
            "score": round(sum(bleu) / 4 / 3 + rouge / 3 + cider / 10 / 3, 4),
        }

    # ---- tag 3: match ----
    items = routed.get(3, [])
    if items:
        f1s = [f for f in (match_result(a, g)[1] for a, g, _ in items) if f == f]
        out["match"] = {
            "n": len(items), "n_with_gt_coords": len(f1s),
            "f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "note": ("GT 에 좌표가 없어 계산 불가 — DriveBench 는 tag3 정답을 "
                     "자연어로 바꿔놨다") if not f1s else None,
        }

    # ---- tag 1: ChatGPT ----
    out["chatgpt"] = {"n": len(routed.get(1, [])), "score": None,
                      "note": "GPT API 필요 — 계산 불가 (공식 최종점수의 40%)"}

    # ---- 최종 점수 (GPT 항 0 으로 둔 하한. 공식 수치가 아니다) ----
    acc = out.get("accuracy", {}).get("official_exact", 0.0)
    lang = out.get("language", {}).get("score", 0.0)
    mf1 = out.get("match", {}).get("f1")
    mt = (mf1 / 2.0) if mf1 is not None else 0.0
    out["final_lower_bound_no_gpt"] = round(0.4 * 0.0 + 0.2 * lang + 0.2 * mt + 0.2 * acc, 4)

    print(f"\n=== {args.label or args.pred} ===")
    a = out.get("accuracy", {})
    print(f"  accuracy (tag0, n={a.get('n')})")
    print(f"    공식 완전일치     {a.get('official_exact')}")
    print(f"    정규화 일치       {a.get('normalized')}")
    print(f"    객관식 문자일치   {a.get('mcq_letter_match')}  (n={a.get('mcq_n')})")
    print(f"    비객관식 정규화   {a.get('nonmcq_normalized')}  (n={a.get('nonmcq_n')})")
    l = out.get("language", {})
    print(f"  language (tag2, n={l.get('n')})  score={l.get('score')}")
    print(f"    BLEU-4 {l.get('BLEU-4')}  ROUGE-L {l.get('ROUGE-L')}  CIDEr {l.get('CIDEr')}")
    m = out.get("match", {})
    print(f"  match (tag3, n={m.get('n')})  f1={m.get('f1')}"
          + (f"  ← {m.get('note')}" if m.get("note") else ""))
    print(f"  chatgpt (tag1, n={out['chatgpt']['n']})  계산 불가")
    print(f"  최종점수 하한(GPT=0): {out['final_lower_bound_no_gpt']}")

    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
