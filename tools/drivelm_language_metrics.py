"""DriveLM 공식 지표 중 언어 지표(BLEU/ROUGE-L/CIDEr)와 accuracy 를 계산한다.

공식 evaluation.py 는 language_evaluation.CocoEvaluator(coco_types=["BLEU",
"ROUGE_L","CIDEr"]) 를 쓰는데, 그 안이 pycocoevalcap 이라 같은 구현을 직접 부른다.

공식과 다른 점 두 가지를 반드시 같이 보고해야 한다:
  - PTBTokenizer 미사용. 공식은 자바 기반 PTBTokenizer 로 먼저 토큰화하는데 이
    컨테이너에는 자바가 없고 비root 라 설치도 안 된다. 여기서는 소문자화 + 구두점
    분리로 근사한다. BLEU/ROUGE 는 영향이 작지만 CIDEr 는 tf-idf 라 더 민감하다.
  - METEOR 없음. 자바가 필수라 계산할 수 없다.

accuracy 는 tools/official_metrics.py 의 규칙(정답 문자열 포함)을 쓰되, 원본
데이터에 공식 tag 가 없어 닫힌 어휘 문항만 고른다 — 공식과 선별 기준이 다르다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.official_metrics import accuracy_hit, is_closed_vocab, match_result  # noqa: E402

_PUNCT = re.compile(r"([.,!?;:()\[\]{}\"])")
_WS = re.compile(r"\s+")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def answer_of(text: str) -> str:
    """모델 출력에서 최종 답변만 꺼낸다."""
    m = _ANSWER.search(text or "")
    if m:
        return m.group(1).strip()
    return _THINK.sub("", text or "").replace("<answer>", "").replace("</answer>", "").strip()


def ptb_like(s: str) -> str:
    """PTBTokenizer 근사: 소문자화 + 구두점 분리 + 공백 정규화."""
    s = _PUNCT.sub(r" \1 ", (s or "").lower())
    return _WS.sub(" ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="blankview 형식 json (per_sample) 또는 {id: text}")
    ap.add_argument("--gt", required=True, help="id/solution 을 가진 데이터 json")
    ap.add_argument("--field", default="full_text", help="per_sample 안에서 쓸 예측 필드")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    raw = json.load(open(args.pred))
    samples = raw["per_sample"] if isinstance(raw, dict) and "per_sample" in raw else raw
    preds = {s["id"]: s[args.field] for s in samples}
    gts = {r["id"]: r["solution"] for r in json.load(open(args.gt))}

    ids = [i for i in preds if i in gts and (gts[i] or "").strip()]
    print(f"예측 {len(preds)} | GT 매칭 {len(ids)}")

    res = {i: [ptb_like(answer_of(preds[i]))] for i in ids}
    ref = {i: [ptb_like(gts[i])] for i in ids}

    bleu, _ = Bleu(4).compute_score(ref, res)
    rouge, _ = Rouge().compute_score(ref, res)
    cider, _ = Cider().compute_score(ref, res)

    # accuracy: 닫힌 어휘 문항만 (공식은 tag 로 고르지만 원본에 tag 가 없다)
    closed = [i for i in ids if is_closed_vocab(gts[i])]
    acc = (sum(accuracy_hit(answer_of(preds[i]), gts[i]) for i in closed) / len(closed)
           if closed else float("nan"))

    # match: GT 에 좌표가 있는 문항만
    f1s = [f for f in (match_result(answer_of(preds[i]), gts[i])[1] for i in ids)
           if f == f]
    match_f1 = sum(f1s) / len(f1s) if f1s else float("nan")

    out = {
        "n": len(ids),
        "BLEU-1": round(bleu[0], 4), "BLEU-2": round(bleu[1], 4),
        "BLEU-3": round(bleu[2], 4), "BLEU-4": round(bleu[3], 4),
        "ROUGE-L": round(rouge, 4), "CIDEr": round(cider, 4),
        "METEOR": None,
        "accuracy": round(acc, 4) if acc == acc else None,
        "accuracy_n": len(closed),
        "match_f1": round(match_f1, 4) if match_f1 == match_f1 else None,
        "match_n": len(f1s),
        "caveats": ["PTBTokenizer 미사용(자바 없음) — 파이썬 근사",
                    "METEOR 미계산(자바 필수)",
                    "accuracy 문항 선별 기준이 공식과 다름(tag 부재)"],
    }
    for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"]:
        print(f"  {k:8s} {out[k]}")
    print(f"  {'METEOR':8s} 계산 불가 (자바 없음)")
    print(f"  {'accuracy':8s} {out['accuracy']}  (닫힌어휘 {out['accuracy_n']}문항)")
    print(f"  {'match_f1':8s} {out['match_f1']}  (좌표 {out['match_n']}문항)")

    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
