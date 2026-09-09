"""DriveLM 공식 평가 지표를 그대로 이식한다.

출처: OpenDriveLab/DriveLM challenge/evaluation.py.
공식 최종 점수는 weights=[0.4, 0.2, 0.2, 0.2] 로 [ChatGPT, Language, Match, Accuracy]
를 섞는데, ChatGPT 항은 GPT-4 API 가 필요하고 Match 도 그 절반을 ChatGPT 로 채운다.
따라서 최종 점수(논문 Average)는 로컬에서 만들 수 없다. 여기서는 API 없이 정확히
재현 가능한 두 항만 계산한다:

  match_f1  공식 match_result() 를 그대로 옮긴 것. 답변에서 소수 좌표쌍을 뽑아
            GT 좌표쌍과 L1 거리 16 이내로 최근접 매칭하고 F1 을 낸다. 논문 Match
            열(49.77)의 ChatGPT 를 뺀 절반에 해당한다.
  accuracy  공식 eval_acc() 와 같은 규칙(정답 문자열 포함 여부). 원본 데이터에
            tag 가 없어 공식처럼 문항을 선별할 수 없으므로, 답이 닫힌 어휘인
            문항만 골라 적용한다 — 선별 기준이 공식과 다르다는 점을 명시해야 한다.
"""

from __future__ import annotations

import re

import numpy as np

# 공식 코드와 동일: 소수점을 가진 수만 좌표로 취급한다
_NUM_RE = re.compile(r"\d+\.\d+")
MATCH_THRESHOLD = 16  # 공식 상수


def match_result(answer: str, gt: str) -> tuple[list, float]:
    """공식 evaluation_suit.match_result 의 축자 이식."""
    answer_nums = _NUM_RE.findall(answer or "")
    gt_nums = _NUM_RE.findall(gt or "")
    if len(answer_nums) % 2 != 0:
        answer_nums = answer_nums[:-1]
    if len(gt_nums) % 2 != 0:
        gt_nums = gt_nums[:-1]
    if not gt_nums:
        return [], float("nan")  # GT 에 좌표가 없으면 이 지표의 대상이 아니다

    answer_arr = np.array([float(x) for x in answer_nums]).reshape(-1, 2)
    gt_arr = np.array([float(x) for x in gt_nums]).reshape(-1, 2)
    length = len(gt_arr)

    matched, tp, fp = [], 0, 0
    for pred in answer_arr:
        closest_distance, closest_gt, closest_id = float("inf"), None, None
        for i, gt_pt in enumerate(gt_arr):
            d = np.sum(np.abs(pred - gt_pt))
            if d < closest_distance:
                closest_distance, closest_gt, closest_id = d, gt_pt, i
        if closest_distance < MATCH_THRESHOLD:
            tp += 1
            matched.append(closest_gt)
            gt_arr = np.delete(gt_arr, closest_id, axis=0)
        else:
            fp += 1
    fn = length - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return matched, f1


# 공식 eval_acc 는 "GT 문자열이 답변에 포함되면 정답" 규칙이다. 닫힌 어휘 문항에만
# 의미가 있으므로, 그런 문항을 식별하는 보수적인 판정을 둔다.
_CLOSED_VOCAB = (
    "yes.", "no.", "going ahead.", "moving.", "stationary.", "turning left.",
    "turning right.", "low.", "high.", "medium.", "back up.", "brake.",
)


def is_closed_vocab(gt: str) -> bool:
    return (gt or "").strip().lower() in _CLOSED_VOCAB


def accuracy_hit(answer: str, gt: str) -> bool:
    """공식 eval_acc 와 같은 포함 판정."""
    return (gt or "").strip().lower() in (answer or "").strip().lower()
