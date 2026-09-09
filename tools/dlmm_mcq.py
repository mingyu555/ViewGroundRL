"""DriveLMM-o1 MCQ 정확도. 공식 evaluation_script.py 의 로직을 그대로 옮긴다.

- MCQ 채점 대상은 idx 마지막 숫자가 [2,3,4,5,8] 인 문항뿐이다 (test 2,391/4,634).
- 최종 답변은 정해진 구분자 목록 중 처음 걸리는 것으로 자른 뒤, 'A)~F)' 첫 매치의
  문자만 비교한다.
GPT 채점이 필요한 나머지 6개 열(Risk Assess./Rule Adh./Scene Aware./Relevance/
Missing/Reason.)은 여기서 계산하지 않는다.
"""
import argparse, json, re

MCQ_QIDX = {2, 3, 4, 5, 8}
SPLITTERS = ["The final answer is:", "**Final Answer:**", "Final Answer", "Answer",
             "Why take this action?:", "**Final Answer**", "**Final Decision**:",
             "Final Step:", "<CONCLUSION>"]
OPT = re.compile(r"([A-F])\)\s*(.+)")


def extract_final_answer(text, lenient=False):
    for opt in SPLITTERS:
        if opt in text:
            return text.split(opt)[-1]
    # 공식 로직은 구분자가 없으면 "" 를 돌려주고, 그러면 정답이라도 자동 오답이 된다.
    # lenient 는 그때 응답 전체에서 선택지 문자를 찾는다 - '형식 준수'와 '정답 여부'를
    # 분리해 보기 위한 것이며 공식 규약이 아니다.
    return text if lenient else ""


def extract_options(text):
    m = OPT.findall(text)
    if m:
        return [(a, b) for a, b in m]
    if "none of the " in text.lower():
        return [("F", "None of the option.")]
    return [("none", "none")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd1/vgrl/data/dlmm_test.json")
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()

    gt = {r["id"].split("::", 1)[1]: r["solution"] for r in json.load(open(args.dataset))}
    preds = json.load(open(args.pred))

    n = hit = no_final = no_opt = 0
    for p in preds:
        idx = p["idx"]
        if int(idx.rsplit("_", 1)[-1]) not in MCQ_QIDX or idx not in gt:
            continue
        n += 1
        g = extract_options(gt[idx])[0][0]
        fa = extract_final_answer(p["llm-response"], args.lenient)
        if not fa:
            no_final += 1
        l = extract_options(fa)[0][0]
        if l == "none":
            no_opt += 1
        if l == g:
            hit += 1
    print(f"MCQ 채점 대상 {n}  ({'lenient' if args.lenient else 'strict(공식)'})")
    print(f"MCQ Accuracy: {hit/n:.2%}   ({hit}/{n})")
    print(f"  최종답변 구분자 없음 {no_final} ({no_final/n:.1%})")
    print(f"  선택지 문자 추출 실패 {no_opt} ({no_opt/n:.1%})")


if __name__ == "__main__":
    main()
