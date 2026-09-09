"""DriveBench 를 task 축(Percep./Predict./Plan./Behav./Avg)으로 정리한다.

DriveBench 논문의 per-task 표는 GPT 채점(README: "Acc, Language, GPT, GPTctx")이라
API 없이는 그대로 재현할 수 없다. 여기서는 task 마다 로컬에서 계산 가능한 지표만
내고, 불가능한 칸은 비운다.

task 와 공식 tag 의 대응 (drivebench-test.json 실측):
  Perception 400 = tag0 200(객관식) + tag2 200(language, 좌표 포함)
  Prediction 261 = tag0 200(자유서술 Yes/No) + tag3  61(match)
  Planning   600 = tag1 600  → ChatGPT 전용, 로컬 계산 불가
  Behavior   200 = tag0 200(객관식)

따라서 Avg 는 Planning 을 뺀 부분 평균이다. 공식 수치가 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.official_metrics import match_result  # noqa: E402
from vgrl.prompt_format import extract_answer  # noqa: E402

_PUNCT = re.compile(r"([.,!?;:()\[\]{}\"])")
_WS = re.compile(r"\s+")
_LETTER = re.compile(r"\b([A-D])\b")
TASKS = ["perception", "prediction", "planning", "behavior"]


def ptb(s):
    return _WS.sub(" ", _PUNCT.sub(r" \1 ", (s or "").lower())).strip()


def norm(s):
    return _WS.sub(" ", re.sub(r"[.,!?;:\s]+$", "", (s or "").strip().lower()))


def letter(s):
    m = _LETTER.search((s or "").strip().upper())
    return m.group(1) if m else None


def lang_score(items):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    res = {str(i): [ptb(a)] for i, (a, _) in enumerate(items)}
    ref = {str(i): [ptb(g)] for i, (_, g) in enumerate(items)}
    b, _ = Bleu(4).compute_score(ref, res)
    r, _ = Rouge().compute_score(ref, res)
    c, _ = Cider().compute_score(ref, res)
    # 공식 language 수식 (evaluation.py 194~212행)
    return sum(b) / 4 / 3 + r / 3 + c / 10 / 3


def score_one(pred_file, meta):
    preds = json.load(open(pred_file))
    buckets = defaultdict(list)      # (task, tag) -> [(ans, gt, is_mcq)]
    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        a = extract_answer(p["llm-response"]) or p["llm-response"]
        mcq = "select the correct answer" in m["question"]
        for t in m["tag"]:
            buckets[(m.get("question_type", m.get("category")), t)].append((a, m["solution"], mcq))

    out = {}
    for task in TASKS:
        d = {"n": 0}
        # tag0 → accuracy
        it = buckets.get((task, 0), [])
        if it:
            d["n"] += len(it)
            d["acc_n"] = len(it)
            if it[0][2]:   # 객관식이면 선택지 문자로 채점
                d["accuracy"] = round(
                    sum(letter(a) == norm(g).upper() for a, g, _ in it) / len(it), 4)
                d["acc_kind"] = "객관식 문자일치"
            else:
                d["accuracy"] = round(sum(norm(a) == norm(g) for a, g, _ in it) / len(it), 4)
                d["acc_kind"] = "정규화 일치"
            d["accuracy_official_exact"] = round(
                sum((a or "").strip() == (g or "").strip() for a, g, _ in it) / len(it), 4)
        # tag2 → language
        it = buckets.get((task, 2), [])
        if it:
            d["n"] += len(it)
            d["lang_n"] = len(it)
            d["language"] = round(lang_score([(a, g) for a, g, _ in it]), 4)
        # tag3 → match
        it = buckets.get((task, 3), [])
        if it:
            d["n"] += len(it)
            d["match_n"] = len(it)
            f1s = [f for f in (match_result(a, g)[1] for a, g, _ in it) if f == f]
            d["match"] = round(sum(f1s) / len(f1s), 4) if f1s else None
            if not f1s:
                d["match_note"] = "GT 에 좌표 없음 → 계산 불가"
        # tag1 → ChatGPT
        it = buckets.get((task, 1), [])
        if it:
            d["n"] += len(it)
            d["chatgpt_n"] = len(it)
            d["chatgpt"] = None
            d["chatgpt_note"] = "GPT API 필요 → 계산 불가"
        out[task] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="라벨=경로 형태")
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/drivebench_test.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = json.load(open(args.dataset))
    meta = {r["id"]: r for r in src}

    all_out = {}
    for spec in args.preds:
        label, _, path = spec.partition("=")
        all_out[label] = score_one(path, meta)

    def cell(d, key):
        v = d.get(key)
        return "—" if v is None else f"{v:.4f}"

    for key, title in [("accuracy", "accuracy (task 별 해당 지표)"),
                       ("language", "language (공식 수식)"),
                       ("match", "match (좌표 F1)")]:
        print(f"\n### {title}")
        print(f"{'모델':26s}" + "".join(f"{t[:9].capitalize():>12s}" for t in TASKS) + f"{'Avg*':>12s}")
        for label, res in all_out.items():
            vals = [res[t].get(key) for t in TASKS]
            got = [v for v in vals if v is not None]
            avg = f"{sum(got)/len(got):.4f}" if got else "—"
            print(f"{label:26s}" + "".join(f"{cell(res[t], key):>12s}" for t in TASKS)
                  + f"{avg:>12s}")

    print("\n* Avg 는 계산 가능한 task 만의 평균이다. Planning 은 전부 ChatGPT 채점")
    print("  대상이라 로컬에서 낼 수 없어 빠진다 — 공식 Avg 가 아니다.")
    print("\ntask 별 문항 구성:")
    r0 = next(iter(all_out.values()))
    for t in TASKS:
        d = r0[t]
        parts = []
        if d.get("acc_n"):     parts.append(f"accuracy {d['acc_n']}({d.get('acc_kind')})")
        if d.get("lang_n"):    parts.append(f"language {d['lang_n']}")
        if d.get("match_n"):   parts.append(f"match {d['match_n']}"
                                            + (f" [{d['match_note']}]" if d.get('match_note') else ""))
        if d.get("chatgpt_n"): parts.append(f"ChatGPT {d['chatgpt_n']} [계산 불가]")
        print(f"  {t:12s} n={d['n']:4d}  " + " + ".join(parts))

    if args.out:
        json.dump(all_out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
