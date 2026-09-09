"""DriveLMM-o1 blank-view 진단. 세 분기의 MCQ 정확도와 텍스트 F1 을 비교한다.

full        : 격자 6셀 전부
mask_ev     : 질문이 지목한 카메라 셀을 검게
mask_ctrl   : 근거뷰가 아닌 셀 하나를 검게 (대조군)

drop = full - mask,  gap = drop_ev - drop_ctrl.
gap 이 0 이면 모델이 근거뷰를 특별히 쓰지 않는다는 뜻이다.
"""
import argparse, json, re
from collections import Counter

MCQ_QIDX = {2, 3, 4, 5, 8}
SPLITTERS = ["The final answer is:", "**Final Answer:**", "Final Answer", "Answer",
             "Why take this action?:", "**Final Answer**", "**Final Decision**:",
             "Final Step:", "<CONCLUSION>"]
OPT = re.compile(r"([A-F])\)\s*(.+)")
WORD = re.compile(r"[a-z0-9]+")


def final(text):
    for o in SPLITTERS:
        if o in text:
            return text.split(o)[-1]
    return text


def letter(text):
    m = OPT.findall(text)
    return m[0][0] if m else "none"


def f1(pred, gold):
    p, g = Counter(WORD.findall(pred.lower())), Counter(WORD.findall(gold.lower()))
    if not p or not g:
        return 0.0
    ov = sum((p & g).values())
    if ov == 0:
        return 0.0
    pr, rc = ov / sum(p.values()), ov / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def score(pred_file, gt):
    preds = json.load(open(pred_file))
    mh = mn = 0
    fs = []
    for x in preds:
        if x["idx"] not in gt:
            continue
        g = gt[x["idx"]]
        fa = final(x["llm-response"])
        fs.append(f1(fa, g))
        if int(x["idx"].rsplit("_", 1)[-1]) in MCQ_QIDX:
            mn += 1
            if letter(fa) == letter(g):
                mh += 1
    return {"n": len(fs), "f1": sum(fs) / max(1, len(fs)),
            "mcq_n": mn, "mcq": mh / max(1, mn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--mask_ev", required=True)
    ap.add_argument("--mask_ctrl", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd1/vgrl/data/dlmm_test_vg.json")
    a = ap.parse_args()
    gt = {r["id"].split("::", 1)[1]: r["solution"] for r in json.load(open(a.dataset))}
    r = {k: score(getattr(a, k), gt) for k in ("full", "mask_ev", "mask_ctrl")}
    print(f"{'분기':10} {'n':>5} {'F1':>8} {'MCQ':>8}  (MCQ n)")
    for k in ("full", "mask_ev", "mask_ctrl"):
        v = r[k]
        print(f"{k:10} {v['n']:>5} {v['f1']:>8.4f} {v['mcq']:>8.4f}  ({v['mcq_n']})")
    print()
    for m, lab in (("f1", "F1"), ("mcq", "MCQ")):
        de = r["full"][m] - r["mask_ev"][m]
        dc = r["full"][m] - r["mask_ctrl"][m]
        print(f"{lab:4}  drop_ev {de:+.4f}   drop_ctrl {dc:+.4f}   gap {de - dc:+.4f}")


if __name__ == "__main__":
    main()
