"""DriveBench GPT-Score 를 로컬 LLM 심판으로 매긴다 (task별 0~100).

DriveBench 논문(arXiv:2501.04003) Figure 22/23 의 루브릭을 축자 이식했다. 원 논문은
GPT-3.5-turbo 를 쓰지만 API 키가 없어 로컬 모델로 대체한다. 평가 대상이 전부 Qwen
계열이므로 자기편향을 피하려고 다른 계열(Mistral)을 심판으로 둔다.

DESC(ground truth object visual description)는 DriveBench 가 GPTctx 라 부르는 맥락
정보다. DriveBench 프레임이 전부 공개 v1_1_train_nus.json 안에 있어서 그 파일의
key_object_infos.Visual_description 으로 만든다.

교정: ReCogDrive(arXiv:2506.08052) Table 7 이 Qwen2-VL 7B 의 clean 점수를
Percep. 28.99 / Predict. 37.89 / Plan. 57.04 / Behav. 49.07 로 발표했다. 우리 심판이
같은 예측에서 그 값을 재현하는지로 신뢰도를 확인한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.prompt_format import extract_answer  # noqa: E402

TASKS = ["perception", "prediction", "planning", "behavior"]

# ---- Figure 22: 객관식 루브릭 (축자) ----
MCQ_RUBRIC = """Please evaluate the predicted answer on a scale from 0 to 100 using the following criteria:

1. Answer Correctness (50 points):
- Award up to 50 points if the selected option matches the correct answer.
- Guideline: Assign 0 points if the selected option is wrong. Partial credit is not given for a wrong option.
2. Object Recognition (10 points):
- Score up to 10 points for correctly recognizing and describing the relevant object(s), including attributes like colors, materials, sizes, or shapes.
- Guideline: Deduct points for any missing, misidentified, or irrelevant objects, particularly if they are crucial to the driving context. Deduct points if any important visual details are missing, incorrect, or overly generalized, especially if they affect comprehension or recognition.
3. Object Location and Orientation (15 points):
- Score up to 5 points for a precise description of the object's location, orientation, or position relative to the ego vehicle.
- Award up to 5 points for acknowledging environmental factors, such as lighting, visibility, and other conditions that influence perception.
- Score up to 5 points based on how well the answer reflects an understanding of situational context, such as obstacles, traffic flow, or potential hazards.
- Guideline: Deduct points for inaccuracies or omissions in spatial information that could affect scene understanding. Deduct points if the answer fails to consider factors impacting object visibility or situational awareness. Deduct points for overlooked or misinterpreted contextual factors that may impact driving decisions.
4. Environmental Condition Awareness (15 points):
- Award up to 15 points if the explanation considers environmental conditions (e.g., weather or sensor limitations) that could impact perception.
- Guideline: Deduct points if relevant environmental conditions are ignored or inadequately addressed.
5. Clarity of Reasoning (10 points):
- Award up to 5 points for clear, logically structured reasoning that is easy to understand.
- Assign up to 5 points for grammatical accuracy and coherent structure.
- Guideline: Deduct points for vague or confusing explanations that hinder comprehension. Deduct points for grammar or syntax issues that impact clarity or logical flow.

Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the multiple-choice question: {QUESTION}
Here is the ground truth object visual description: {DESC}
Here is the correct answer: {GT}
Here is the predicted answer and explanation (if any): {PRED}

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (50 points):
2. Object Recognition (10 points):
3. Object Location and Orientation (15 points):
4. Environmental Condition Awareness (15 points):
5. Clarity of Reasoning (10 points):
Total Score:
Brief Summary:"""

# ---- Figure 23: 자유서술 루브릭 (축자) ----
OPEN_RUBRIC = """Please evaluate the predicted answer on a scale from 0 to 100 using the following criteria:

1. Action Alignment (20 points):
- Award up to 20 points based on how closely the predicted action aligns with the correct answer.
- Guideline: Award full points only for exact matches or highly similar actions. Deduct points for any inaccuracies or missing elements. Assign 0 points if no action prediction is provided.
2. Motion Precision (20 points):
- Award up to 20 points based on how closely the predicted motion (e.g., speed up, decelerate) aligns with the correct motion in the answer.
- Guideline: Deduct points if the predicted motion fails to match the type or intensity of the correct answer. Ensure that the intended speed or deceleration aligns accurately with the driving context. Assign 0 points if no motion prediction is provided.
3. Driving Context Appropriateness (15 points):
- Score up to 15 points for the relevance of the predicted answer to the driving context implied by the correct answer, emphasizing logical alignment with the situation.
- Guideline: Award higher scores only if the answer fully reflects an accurate understanding of the driving context. Deduct points if the action or motion is illogical or does not align with the scenario's requirements.
4. Situational Awareness (15 points):
- Award up to 15 points for demonstrated awareness of environmental factors (e.g., traffic participants, obstacles) relevant to the action or motion.
- Guideline: Deduct points if the answer misses key situational details that may lead to unsafe or incorrect predictions.
5. Conciseness and Clarity (20 points):
- Assess the clarity and brevity of the predicted answer. Answers should be concise, clear, and easy to understand, effectively communicating the intended actions and motions.
- Guideline: Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension.
6. Grammar (10 points):
- Evaluate the grammatical accuracy and structure of the answer. Assign up to 5 points for clarity and logical flow, and up to 5 points for grammatical accuracy.
- Guideline: Deduct points for grammar or syntax issues that reduce readability or coherence.

Here is the predicted answer: {PRED}
Here is the correct answer: {GT}

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Action Alignment (20 points):
2. Motion Precision (20 points):
3. Driving Context Appropriateness (15 points):
4. Situational Awareness (15 points):
5. Conciseness and Clarity (20 points):
6. Grammar (10 points):
Total Score:
Brief Summary:"""

_TOTAL = re.compile(r"Total\s*Score\s*[:\-]?\s*\**\s*(\d{1,3})", re.I)
_ANY = re.compile(r"\b(\d{1,3})\s*/\s*100\b")
# 항목별 점수: "1. Answer Correctness (50 points): 40" 또는 "... : 40/50"
_ITEM = re.compile(r"^\s*\**\s*(\d)\.\s*[^:\n]{3,60}?\((\d{1,3})\s*points?\)\s*\**\s*:\s*\**\s*(\d{1,3})",
                   re.I | re.M)


def parse_total(txt: str):
    """총점을 뽑는다. 심판이 총점 줄까지 못 가고 잘린 경우 항목 점수를 더한다."""
    txt = txt or ""
    m = _TOTAL.search(txt)
    if m:
        return min(int(m.group(1)), 100)
    m = _ANY.search(txt)
    if m:
        return min(int(m.group(1)), 100)
    # 폴백: 항목별 점수 합산. 루브릭 항목이 다 나왔을 때만 인정한다
    items = _ITEM.findall(txt)
    if items:
        seen = {}
        for idx, cap, got in items:
            seen[int(idx)] = min(int(got), int(cap))
        # 객관식 5항목 / 자유서술 6항목
        if len(seen) >= 5:
            return min(sum(seen.values()), 100)
    return None


def build_desc(frame_token, question, f2info) -> str:
    """질문이 지목한 객체의 GT 시각 묘사. DriveBench 가 GPTctx 라 부르는 맥락."""
    info = f2info.get(frame_token) or {}
    if not info:
        return "N/A"
    # 질문 안의 <cN,CAM,x,y> 와 key 를 카메라+번호로 느슨하게 맞춘다 (좌표계가 다르다:
    # DriveBench 는 정규화, key_object_infos 는 픽셀)
    want = re.findall(r"<(c\d+),(CAM_[A-Z_]+),", question or "")
    parts = []
    for cid, cam in want:
        for k, v in info.items():
            if k.startswith(f"<{cid},{cam},"):
                parts.append(f"{cid} ({cam}): {v.get('Category','?')}, "
                             f"{v.get('Status','?')}, {v.get('Visual_description','?')}")
                break
    if not parts:      # 지목 객체가 없으면 프레임 전체 객체를 넣는다
        for k, v in list(info.items())[:6]:
            parts.append(f"{k.split(',')[0].lstrip('<')}: {v.get('Category','?')}, "
                         f"{v.get('Status','?')}, {v.get('Visual_description','?')}")
    return "; ".join(parts) if parts else "N/A"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd4/mingyu/vgrl/data/drivebench_test.json")
    ap.add_argument("--drivelm_raw", default="/mnt/ssd4/mingyu/vgrl/raw/v1_1_train_nus.json")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rubric_by_mcq", action="store_true",
                    help="객관식 여부로 루브릭을 고른다 (기본은 task 기준)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    meta = {r["id"]: r for r in json.load(open(args.dataset))}
    raw = json.load(open(args.drivelm_raw))
    f2info = {fk: fv.get("key_object_infos", {})
              for sv in raw.values() for fk, fv in sv["key_frames"].items()}

    preds = json.load(open(args.pred))
    if args.limit:
        preds = preds[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.judge)
    jobs = []
    for p in preds:
        m = meta.get(p["id"])
        if m is None:
            continue
        pred = extract_answer(p["llm-response"]) or p["llm-response"]
        task = m.get("question_type", m.get("category"))
        is_mcq = "select the correct answer" in m["question"]
        # 루브릭 라우팅. 논문은 "Rubrics are adapted for each specific task and
        # question type" 이라고만 하고 부록에 2종만 공개했다. 내용으로 맞춘다:
        #   객체 루브릭(Fig 22) = 질문이 특정 객체를 지목하는 경우
        #   행동 루브릭(Fig 23) = Action Alignment / Motion Precision 이 항목이므로
        #                          ego 의 행동·속도를 묻는 behavior / planning
        # 교정 근거: planning 을 행동 루브릭으로 채점하니 ReCogDrive 발표값과
        # 57.04 vs 56.32 (-0.72) 로 거의 일치했다.
        use_object_rubric = (task in ("perception", "prediction")) if not args.rubric_by_mcq \
            else is_mcq
        if use_object_rubric:
            body = MCQ_RUBRIC.format(QUESTION=m["question"],
                                     DESC=build_desc(m["frame_token"], m["question"], f2info),
                                     GT=m["solution"], PRED=pred)
        else:
            body = OPEN_RUBRIC.format(PRED=pred, GT=m["solution"])
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        jobs.append((task, is_mcq, text))

    print(f"채점 대상 {len(jobs)}건 (심판: {os.path.basename(args.judge)})", flush=True)

    llm = LLM(model=args.judge, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=4096, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate([t for _, _, t in jobs], sp)

    per_task = defaultdict(list)
    unparsed = 0
    truncated = 0
    samples = []
    for (task, is_mcq, _), o in zip(jobs, outs):
        txt = o.outputs[0].text
        if o.outputs[0].finish_reason == "length":
            truncated += 1
        v = parse_total(txt)
        if v is None:
            unparsed += 1
            if len(samples) < 3:
                samples.append({"task": task, "mcq": is_mcq,
                                "finish": o.outputs[0].finish_reason,
                                "tail": txt[-600:]})
            continue
        per_task[task].append(v)

    res = {"label": args.label, "judge": args.judge, "pred_file": args.pred,
           "n": len(jobs), "unparsed": unparsed, "truncated": truncated,
           "unparsed_samples": samples, "per_task": {}}
    for t in TASKS:
        v = per_task[t]
        res["per_task"][t] = {"n": len(v),
                              "gpt_score": round(sum(v) / len(v), 2) if v else None}
    got = [res["per_task"][t]["gpt_score"] for t in TASKS
           if res["per_task"][t]["gpt_score"] is not None]
    res["avg"] = round(sum(got) / len(got), 2) if got else None

    print(f"\n=== {args.label or args.pred} ===")
    print("  " + "".join(f"{t[:9].capitalize():>12s}" for t in TASKS) + f"{'Avg':>12s}")
    print("  " + "".join(
        f"{(res['per_task'][t]['gpt_score'] if res['per_task'][t]['gpt_score'] is not None else '—'):>12}"
        for t in TASKS) + f"{(res['avg'] if res['avg'] is not None else '—'):>12}")
    print(f"  파싱 실패 {unparsed} / {len(jobs)}  (길이초과로 잘림 {truncated})")
    for smp in samples[:2]:
        print(f"    [{smp['task']}, mcq={smp['mcq']}, finish={smp['finish']}] "
              f"...{smp['tail'][-220:]!r}")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"  저장 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
