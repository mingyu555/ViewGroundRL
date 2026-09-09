"""근거뷰/대조뷰를 가렸을 때의 실제 KL 을 직접 측정한다 (학습 없이).

blank-view 진단은 성능 낙폭으로 접지를 재는데, 학습 전 모델은 정답률이 0 에 가까워
낙폭이 0-0=0 으로 무의미해진다. KL 은 정답 여부와 무관하게 "출력 분포가 얼마나
흔들리는가"를 재므로 그 구간에서도 측정된다.

학습 시 관점항이 쓰는 것과 같은 양을 계산한다:
    kl1 = KL( p(y | 전체 6뷰) || p(y | 근거뷰 가림) )
    kl2 = KL( p(y | 전체 6뷰) || p(y | 대조뷰 가림) )
y 는 정답(teacher forcing)을 쓴다 — 학습 때는 롤아웃을 쓰지만, 모델 간 비교에는
같은 y 를 고정하는 편이 교란이 없다. k3 추정기(r - log r - 1)로 토큰별 평균.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/nuins_eval_vg.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max_pixels", type=int, default=401408)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # 데이터셋마다 하위 과제/카테고리 축이 달라, 평균 하나로는 "어느 과제에서
    # 접지가 살아있나"가 보이지 않는다. 그리고 질문에 카메라/방향이 적혀 있으면
    # 그 뷰를 가리는 것만으로 정답이 무너져 margin 이 과대평가된다 (NuInstruct 에서
    # 실제로 그랬다). 두 축을 모두 쪼개서 낸다.
    ap.add_argument("--group_by", default="", help="task | category | (빈값=전체만)")
    ap.add_argument("--hint_split", action="store_true",
                    help="질문이 카메라/방향을 이미 알려주는지로 쪼갠다")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.load(open(args.dataset))
    rows = [r for r in rows if r.get("vg_usable")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                         max_pixels=args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()

    def logprobs(images, prompt_msgs, target):
        """정답 토큰들의 로그확률 [T]."""
        text = proc.apply_chat_template(prompt_msgs, tokenize=False,
                                        add_generation_prompt=True)
        full = text + target
        enc_p = proc(text=[text], images=[images], return_tensors="pt")
        enc = proc(text=[full], images=[images], return_tensors="pt").to("cuda")
        n_prompt = enc_p["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**enc)
        lg = out.logits[0, n_prompt - 1 : -1].float().log_softmax(-1)
        tgt = enc["input_ids"][0, n_prompt:]
        if tgt.numel() == 0:
            return None
        return lg.gather(-1, tgt[:, None]).squeeze(-1), tgt

    # 뷰와 무관한 토큰(관사/전치사/구두점/정형구)은 두 분기에 거의 같은 값을 보태
    # margin 에서 상쇄되지만, 평균을 희석해 크기를 줄인다. 정답 길이가 데이터셋마다
    # 달라(NuInstruct 33자 vs OmniDrive 90토큰) 데이터셋 간 비교가 특히 오염된다.
    # 세 가지 평균을 한 번의 순전파로 같이 낸다.
    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
        "and", "or", "but", "that", "this", "these", "those", "it", "its",
        "there", "here", "as", "if", "than", "then", "so", "which", "you",
        "your", "we", "our", "us", "will", "would", "should", "could", "can",
        "may", "might", "must", "have", "has", "had", "do", "does", "did",
        "not", "no", "also", "while", "however", "additionally", "firstly",
        "scenario", "following", "given", "current", "vehicle", "ego",
    }
    PUNCT = set(".,;:!?()[]{}<>-—–\"'`/\\|+*=&%$#@~^_")

    def token_kinds(ids):
        """토큰별 (내용어인가, 접지스팬인가). 접지스팬은 데이터셋 형식으로 판별한다."""
        toks = [proc.tokenizer.decode([int(t)]) for t in ids]
        content = []
        for t in toks:
            w = t.strip().lower()
            content.append(bool(w) and w not in STOP
                           and not all(c in PUNCT or c.isspace() for c in w))
        return toks, content

    # 접지 스팬: NuInstruct 는 <class>[cN,...], OmniDrive 는 (+x, +y)
    SPAN_RE = [re.compile(r"<[^>]{1,40}>\s*\[[^\]]{0,80}\]"),
               re.compile(r"\(\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\)")]

    def span_mask(toks, target):
        """토큰을 누적 문자열 위치로 되짚어 접지 스팬과 겹치는지 본다."""
        spans = [m.span() for rx in SPAN_RE for m in rx.finditer(target)]
        if not spans:
            return [False] * len(toks)
        out, pos = [], 0
        for t in toks:
            a, b = pos, pos + len(t)
            out.append(any(a < e and b > s0 for s0, e in spans))
            pos = b
        return out

    def blackout(im):
        return Image.new("RGB", im.size, (0, 0, 0))

    CAM_RE = re.compile(
        r"CAM_(?:FRONT|BACK)(?:_(?:LEFT|RIGHT))?|"
        r"\b(?:front|back|rear|behind|left|right|forward)\b", re.I)

    def qtext(r):
        if r.get("question"):
            return r["question"]
        if r.get("prompt_text"):
            return r["prompt_text"]
        for m in r.get("prompt") or []:
            if m.get("role") == "user":
                return " ".join(c.get("text", "") for c in m["content"])
        return ""

    recs, k1s, k2s, ok = [], [], [], 0
    for i, r in enumerate(rows):
        ev = r["evidence_views"]
        pool = [v for v in range(len(r["images"])) if v not in set(ev)]
        if not ev or not pool:
            continue
        e = random.Random(f"{r['id']}:e").choice(ev)
        c = random.Random(f"{r['id']}:c").choice(pool)

        drops, drops_c, drops_s = [], [], []
        base = [Image.open(p).convert("RGB") for p in r["images"]]
        tgt = r["solution"]
        got = logprobs(base, r["prompt"], tgt)
        if got is None:
            continue
        lp_full, tgt_ids = got
        toks, is_content = token_kinds(tgt_ids)
        is_span = span_mask(toks, tgt)
        for cell, store in ((e, k1s), (c, k2s)):
            v = list(base)
            v[cell] = blackout(v[cell])
            got2 = logprobs(v, r["prompt"], tgt)
            if got2 is None or got2[0].shape != lp_full.shape:
                store.append(float("nan"))
                drops.append(float("nan")); drops_c.append(float("nan"))
                drops_s.append(float("nan"))
                continue
            lp = got2[0]
            # k3 추정기: r = p_masked / p_full, KL ~= r - log r - 1
            logr = lp - lp_full
            rr = logr.exp()
            store.append(float((rr - logr - 1).mean()))
            d = -logr
            drops.append(float(d.mean()))
            mc = torch.tensor(is_content, device=d.device)[: d.shape[0]]
            ms = torch.tensor(is_span, device=d.device)[: d.shape[0]]
            drops_c.append(float(d[mc].mean()) if bool(mc.any()) else float("nan"))
            drops_s.append(float(d[ms].mean()) if bool(ms.any()) else float("nan"))
        recs.append({"id": r["id"],
                     "group": str(r.get(args.group_by)) if args.group_by else "all",
                     "hint": bool(CAM_RE.search(qtext(r))),
                     "n_ev": len(ev),
                     "kl1": k1s[-1], "kl2": k2s[-1],
                     "d1": drops[0] if len(drops) > 1 else float("nan"),
                     "d2": drops[1] if len(drops) > 1 else float("nan"),
                     "c1": drops_c[0] if len(drops_c) > 1 else float("nan"),
                     "c2": drops_c[1] if len(drops_c) > 1 else float("nan"),
                     "s1": drops_s[0] if len(drops_s) > 1 else float("nan"),
                     "s2": drops_s[1] if len(drops_s) > 1 else float("nan"),
                     "n_tok": len(toks),
                     "n_content": sum(is_content), "n_span": sum(is_span)})
        ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {ok}건 처리", flush=True)

    k1 = [x for x in k1s if not math.isnan(x)]
    k2 = [x for x in k2s if not math.isnan(x)]
    m1 = sum(k1) / len(k1) if k1 else float("nan")
    m2 = sum(k2) / len(k2) if k2 else float("nan")
    res = {"label": args.label, "model": args.model, "n": ok,
           "kl1_evidence": round(m1, 5), "kl2_control": round(m2, 5),
           "margin": round(m1 - m2, 5),
           "ratio": round(m1 / m2, 3) if m2 else None,
           "norm_margin": round((m1 - m2) / (m1 + m2), 4) if (m1 + m2) else None}
    print(f"\n=== {args.label or args.model}  (n={ok})")
    print(f"  kl1 (근거뷰 가림) {m1:.5f}")
    print(f"  kl2 (대조뷰 가림) {m2:.5f}")
    print(f"  margin = kl1-kl2  {m1-m2:+.5f}   비율 {m1/m2 if m2 else float('nan'):.2f}배")
    print(f"  정규화 여유 (kl1-kl2)/(kl1+kl2)  {(m1-m2)/(m1+m2):+.4f}")
    def pair(ka, kb, label):
        a = [x[ka] for x in recs if not math.isnan(x[ka])]
        b = [x[kb] for x in recs if not math.isnan(x[kb])]
        if not a or not b:
            print(f"     {label:14s} (해당 토큰 없음)")
            return
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        res[f"drop_{label}"] = [round(ma, 5), round(mb, 5), round(ma - mb, 5)]
        print(f"     {label:14s} n={len(a):3d}  근거뷰 {ma:+9.5f}  대조뷰 {mb:+9.5f}  "
              f"차이 {ma-mb:+9.5f}")

    nt = [x["n_tok"] for x in recs]
    nc = [x["n_content"] for x in recs]
    ns = [x["n_span"] for x in recs]
    if nt:
        print(f"  토큰 구성: 전체 {sum(nt)/len(nt):.1f}  내용어 {sum(nc)/len(nc):.1f} "
              f"({100*sum(nc)/max(sum(nt),1):.0f}%)  접지스팬 {sum(ns)/len(ns):.1f} "
              f"({100*sum(ns)/max(sum(nt),1):.0f}%)")
    print("  [유계] 토큰 선택별 정답 로그확률 낙폭 nats/token")
    pair("d1", "d2", "전체토큰")
    pair("c1", "c2", "내용어만")
    pair("s1", "s2", "접지스팬만")

    mk1 = sorted(x for x in k1 if not math.isnan(x))
    mk2 = sorted(x for x in k2 if not math.isnan(x))
    if mk1 and mk2:
        print(f"  [중앙값] k3  근거뷰 {mk1[len(mk1)//2]:.5f}  "
              f"대조뷰 {mk2[len(mk2)//2]:.5f}")
    def block(title, rows):
        if not rows:
            return
        print(f"\n  --- {title} 별")
        print(f"  {'':22s} {'n':>5s} {'kl1':>9s} {'kl2':>9s} {'margin':>9s} "
              f"{'비율':>8s} {'정규화':>8s}")
        for key, sub in sorted(rows.items()):
            a = [x["kl1"] for x in sub if not math.isnan(x["kl1"])]
            b = [x["kl2"] for x in sub if not math.isnan(x["kl2"])]
            if not a or not b:
                continue
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            rat = f"{ma/mb:8.2f}" if mb else "     inf"
            nm = (ma - mb) / (ma + mb) if (ma + mb) else float("nan")
            print(f"  {str(key)[:22]:22s} {len(sub):5d} {ma:9.5f} {mb:9.5f} "
                  f"{ma-mb:+9.5f} {rat} {nm:+8.4f}")

    def bucket(fn):
        out = {}
        for x in recs:
            out.setdefault(fn(x), []).append(x)
        return out

    if args.group_by:
        block(args.group_by, bucket(lambda x: x["group"]))
    if args.hint_split:
        block("질문에 카메라/방향 언급", bucket(lambda x: "있음" if x["hint"] else "없음"))
    block("근거뷰 개수", bucket(lambda x: 1 if x["n_ev"] == 1 else 2))

    res["rows"] = recs
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
