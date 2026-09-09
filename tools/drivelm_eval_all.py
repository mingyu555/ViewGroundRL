"""DriveLM 예측에 대해 돌릴 수 있는 지표를 전부 계산한다 (카테고리별 + 전체).

  f1        토큰 중첩 F1 (vgrl.rewards._f1) — 우리가 학습 내내 써온 지표
  obj       객체 참조 보상 (vgrl.rewards.object_reference_reward)
  accuracy  공식 eval_acc 규칙(정답 문자열 포함). 원본에 tag 가 없어 닫힌 어휘
            문항만 고른다 — 선별 기준이 공식과 다르다
  match_f1  공식 match_result. 소수 좌표쌍을 L1 16 이내로 매칭한 F1
  BLEU-1~4 / ROUGE-L / CIDEr   pycocoevalcap (공식 language_evaluation 의 내부)

계산 불가:
  METEOR       자바 필수인데 컨테이너에 없고 비root 라 설치도 막힌다
  ChatGPT score  GPT API 필요
공식과 다른 점:
  PTBTokenizer 미사용(자바). 소문자화+구두점 분리로 근사한다. CIDEr 는 tf-idf 라
  이 차이에 특히 민감하므로 공식 수치와 직접 비교하면 안 된다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.official_metrics import accuracy_hit, is_closed_vocab, match_result  # noqa: E402
from vgrl.prompt_format import extract_answer  # noqa: E402
from vgrl.rewards import _f1, object_reference_reward  # noqa: E402

_PUNCT = re.compile(r"([.,!?;:()\[\]{}\"])")
_WS = re.compile(r"\s+")
CATS = ["perception", "prediction", "planning", "behavior"]


def ptb_like(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(r" \1 ", (s or "").lower())).strip()


def _norm(s: str) -> str:
    """대소문자·꼬리 구두점·공백 차이를 없앤다. GT 는 "No." 인데 모델은 "no" 라고
    답하는 식의 표기 차이를 정답 판정에서 빼기 위한 것."""
    return _WS.sub(" ", re.sub(r"[.,!?;:\s]+$", "", (s or "").strip().lower()))


def block(items) -> dict:
    """items: [(pred_answer, gt)]"""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    if not items:
        return {}
    preds = [a for a, _ in items]
    gts = [g for _, g in items]

    f1 = sum(_f1(a, g) for a, g in items) / len(items)
    obj = sum(object_reference_reward([a], solution=[g])[0] for a, g in items) / len(items)

    # accuracy 를 세 가지로 낸다. 공식 eval_acc 는 answer == GT 완전 일치라
    # 미학습 모델은 구조적으로 0 에 가깝다 (GT "No." 에 모델이 "no" 라고 답해도 오답).
    # 모델 간 비교에는 정규화 버전이 실제 정보를 준다.
    closed = [(a, g) for a, g in items if is_closed_vocab(g)]
    acc_off = (sum((a or "").strip() == (g or "").strip() for a, g in closed) / len(closed)
               if closed else None)
    acc_norm = (sum(_norm(a) == _norm(g) for a, g in closed) / len(closed)
                if closed else None)
    acc_in = (sum(accuracy_hit(a, g) or _norm(g) in _norm(a) for a, g in closed) / len(closed)
              if closed else None)

    f1s = [f for f in (match_result(a, g)[1] for a, g in items) if f == f]
    match_f1 = sum(f1s) / len(f1s) if f1s else None

    res = {str(i): [ptb_like(p)] for i, p in enumerate(preds)}
    ref = {str(i): [ptb_like(g)] for i, g in enumerate(gts)}
    bleu, _ = Bleu(4).compute_score(ref, res)
    rouge, _ = Rouge().compute_score(ref, res)
    cider, _ = Cider().compute_score(ref, res)

    # 공식 합성 지표. evaluation.py 의 최종 점수 계산부(194~222행) 그대로.
    #   language = mean(BLEU1~4)/3 + ROUGE_L/3 + CIDEr/10/3   (단순 평균이 아니다)
    #   match    = (좌표F1*100 + ChatGPT) / 2 / 100 — GPT 가 없으면 F1 쪽 절반만,
    #              즉 f1/2 가 되고 나머지 절반은 빈 채로 남는다
    language = round(sum(bleu) / 4.0 / 3.0 + rouge / 3.0 + cider / 10.0 / 3.0, 4)
    match_half = round(match_f1 / 2.0, 4) if match_f1 is not None else None

    return {
        "n": len(items),
        "language": language,
        "match_half": match_half,
        "f1": round(f1, 4), "obj": round(obj, 4),
        "accuracy": round(acc_norm, 4) if acc_norm is not None else None,
        "accuracy_official": round(acc_off, 4) if acc_off is not None else None,
        "accuracy_contains": round(acc_in, 4) if acc_in is not None else None,
        "accuracy_n": len(closed),
        "match_f1": round(match_f1, 4) if match_f1 is not None else None,
        "match_n": len(f1s),
        "BLEU-1": round(bleu[0], 4), "BLEU-2": round(bleu[1], 4),
        "BLEU-3": round(bleu[2], 4), "BLEU-4": round(bleu[3], 4),
        "ROUGE-L": round(rouge, 4), "CIDEr": round(cider, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = json.load(open(args.pred))
    by = defaultdict(list)
    allitems = []
    for r in rows:
        a = extract_answer(r["llm-response"]) or r["llm-response"]
        g = r["solution"]
        if not (g or "").strip():
            continue
        by[r["category"]].append((a, g))
        allitems.append((a, g))

    out = {"label": args.label, "pred_file": args.pred,
           "per_category": {}, "overall": block(allitems),
           "not_computed": ["METEOR (자바 없음)", "ChatGPT score (API 필요)",
                            "공식 최종점수 (GPT 가 50% 차지: chatgpt 40% + match 의 절반 10%)"],
           "caveats": ["PTBTokenizer 미사용 — 파이썬 근사, CIDEr 특히 민감",
                       "accuracy 문항 선별 기준이 공식과 다름(tag 부재)"]}
    for c in CATS:
        if by[c]:
            out["per_category"][c] = block(by[c])

    # 공식 최종 점수는 GPT 가 50%(chatgpt 40% + match 의 절반 10%)를 차지해 만들 수
    # 없다. GPT 항을 0 으로 둔 하한만 참고로 남긴다 — 공식 수치가 아니다.
    o = out["overall"]
    if o.get("accuracy_official") is not None:
        out["final_lower_bound_no_gpt"] = round(
            0.4 * 0.0
            + 0.2 * o["language"]
            + 0.2 * (o["match_half"] or 0.0)
            + 0.2 * o["accuracy_official"], 4)

    cols = ["n", "f1", "accuracy", "acc_off", "language", "match_half",
            "match_f1", "BLEU-4", "ROUGE-L", "CIDEr"]
    print(f"\n=== {args.label or args.pred} ===")
    print("  " + "".join(f"{c:>10s}" for c in ["category"] + cols))
    for c in CATS + ["overall"]:
        d = out["overall"] if c == "overall" else out["per_category"].get(c)
        if not d:
            continue
        print("  " + f"{c:>10s}" + "".join(
            f"{('-' if d.get({'acc_off':'accuracy_official'}.get(k,k)) is None else d[{'acc_off':'accuracy_official'}.get(k,k)]):>10}"
            for k in cols))

    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
