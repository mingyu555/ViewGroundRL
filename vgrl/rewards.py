"""Reward functions for DriveLM GRPO.

DriveLM answers are free-form text, so there is no single verifiable signal. The
task splits into three shapes, and the reward routes on the QA category:

  behavior   - a closed vocabulary ("the ego vehicle is going straight. The
               speed is slow") -> exact match on the (steer, speed) pair
  perception / prediction / planning
             - open text -> lexical overlap with the reference, plus credit for
               naming the right object ids / cameras
  multiple choice (DriveLM marks these with "Please select the best answer")
             - exact option match

Every function takes `(completions, **kwargs)` where kwargs carries the dataset
columns, per TRL's reward API, and returns one float per completion.
"""

from __future__ import annotations

import math

import math
import re
from collections import Counter

from .prompt_format import extract_answer, has_format
from .views import CAMERAS, OBJECT_TAG_RE

STEER_VOCAB = ("going straight", "turning left", "turning right")
SPEED_VOCAB = ("fast", "slow", "normal", "stopped", "quickly", "slowly")
OPTION_RE = re.compile(r"\b([A-D])\b")
OBJ_ID_RE = re.compile(r"<(c\d+)\s*,")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(text))


def _f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    overlap = Counter(p) & Counter(g)
    n = sum(overlap.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def _extract_phrases(text: str, vocab: tuple[str, ...]) -> set[str]:
    t = _norm(text)
    return {v for v in vocab if v in t}


def behavior_reward(completions, solution=None, category=None, **kwargs) -> list[float]:
    """1.0 when both the steering and the speed phrase match the reference."""
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        text = _completion_text(comp)
        gold = solution[i] if i < len(solution) else ""
        if category is not None and i < len(category) and category[i] != "behavior":
            out.append(0.0)
            continue
        steer_ok = _extract_phrases(text, STEER_VOCAB) == _extract_phrases(gold, STEER_VOCAB)
        speed_ok = _extract_phrases(text, SPEED_VOCAB) == _extract_phrases(gold, SPEED_VOCAB)
        out.append(1.0 if (steer_ok and speed_ok) else 0.0)
    return out


def text_overlap_reward(completions, solution=None, **kwargs) -> list[float]:
    """Token-level F1 against the reference answer."""
    solution = solution or []
    return [
        _f1(_completion_text(c), solution[i] if i < len(solution) else "")
        for i, c in enumerate(completions)
    ]


def object_reference_reward(completions, solution=None, **kwargs) -> list[float]:
    """Credit for naming the same object ids and cameras as the reference.

    This is the reward most directly aligned with the view-grounding term: an
    answer can only cite the right `<c1,CAM_BACK,...>` if it looked at CAM_BACK.
    Returns 0.0 for references that cite nothing, so those samples neither help
    nor hurt.
    """
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        text = _completion_text(comp)
        gold = solution[i] if i < len(solution) else ""

        gold_ids = set(OBJ_ID_RE.findall(gold))
        gold_cams = set(OBJECT_TAG_RE.findall(gold))
        if not gold_ids and not gold_cams:
            out.append(0.0)
            continue

        pred_ids = set(OBJ_ID_RE.findall(text))
        pred_cams = set(OBJECT_TAG_RE.findall(text))

        parts = []
        if gold_ids:
            parts.append(len(gold_ids & pred_ids) / len(gold_ids | pred_ids or {1}))
        if gold_cams:
            parts.append(len(gold_cams & pred_cams) / len(gold_cams | pred_cams or {1}))
        out.append(sum(parts) / len(parts))
    return out


def multiple_choice_reward(completions, solution=None, **kwargs) -> list[float]:
    """Exact option-letter match for DriveLM's multiple-choice questions."""
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        gold = OPTION_RE.findall((solution[i] if i < len(solution) else "").upper())
        pred = OPTION_RE.findall(_completion_text(comp).upper())
        if not gold:
            out.append(0.0)
        else:
            out.append(1.0 if (pred and pred[0] == gold[0]) else 0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    """Shaping term for the `<think>...</think><answer>...</answer>` contract that
    SFT taught: full credit for well-formed and reasonably sized, half for a usable
    answer without the tags, zero for empty or runaway."""
    out = []
    for comp in completions:
        raw = _raw_text(comp)
        n = len(_tokens(extract_answer(raw)))
        if n == 0 or n > 300:
            out.append(0.0)
        elif has_format(raw):
            out.append(1.0)
        else:
            out.append(0.5)
    return out


def _raw_text(completion) -> str:
    """The generation as produced, tags included.

    TRL passes plain strings for standard datasets and message lists for
    conversational ones.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[-1].get("content", "")
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""
    return ""


def _completion_text(completion) -> str:
    """The part of the generation that should be scored against the reference.

    SFT teaches `<think>reasoning</think><answer>final</answer>`, so every content
    reward reads only `<answer>` — otherwise the reasoning text inflates token
    overlap and a model could score well by restating the question at length.
    Generations that never opened the tag fall back to the tail (see
    prompt_format.extract_answer), so a format slip costs format_reward rather
    than silently zeroing the content rewards.
    """
    return extract_answer(_raw_text(completion))



# ---------------------------------------------------------------------------
# Trajectory planning (nuScenes). DriveLM's rewards are text-shaped; a waypoint
# list needs a geometric one — `text_overlap` on "[(0.00,-0.00), ...]" would score
# digit overlap, which is noise, and `object_reference` is identically 0 because a
# trajectory names no objects.
# ---------------------------------------------------------------------------

WAYPOINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
N_WAYPOINTS = 6
L2_TAU = 2.0
"""Scale of the L2 -> reward map. exp(-L2/2) gives 0.78 at 0.5 m, 0.61 at 1 m,
0.22 at 3 m — a useful gradient across the range our SFT model actually occupies
(UniAD L2 1.01, ST-P3 0.54)."""


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in WAYPOINT_RE.findall(text or "")]


def trajectory_l2_reward(completions, solution=None, **kwargs) -> list[float]:
    """exp(-mean L2 / tau) against the ground-truth waypoints.

    Returns 0.0 when the completion does not yield the expected number of
    waypoints, so a malformed trajectory is penalised rather than partially
    credited — the evaluation harness treats an unparseable prediction as a
    failure too.
    """
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        pred = parse_waypoints(_completion_text(comp))
        gold = parse_waypoints(solution[i] if i < len(solution) else "")
        if len(gold) < N_WAYPOINTS or len(pred) < N_WAYPOINTS:
            out.append(0.0)
            continue
        pred, gold = pred[:N_WAYPOINTS], gold[:N_WAYPOINTS]
        d = [((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 for (px, py), (gx, gy) in zip(pred, gold)]
        out.append(math.exp(-(sum(d) / len(d)) / L2_TAU))
    return out


# ---------------------------------------------------------------------------
# Physics-grounded trajectory reward (AutoDrive-R2, arXiv:2509.01944).
#
# Why this replaces `trajectory_l2_reward`: that reward had a single term
# (position), which gives GRPO nothing to distinguish a physically sensible
# trajectory from a jittery one with the same endpoint error. The measured outcome
# was a policy that moved 0.20 m per trajectory without improving L2 — pushed, but
# in no useful direction.
#
# (Masking is supported but turned out not to matter for training: MindDriver's
# train split is already filtered to fully-observed futures — 0 of 23,388 frames
# have an unobserved step, and `gt_ego_fut_masks` agrees with the released
# `gt_traj_mask.pkl` on all 6,019 val frames. The 18.3% of targets with repeated
# waypoints are genuinely stationary vehicles, not padding. The mask argument is
# kept because the val protocol uses it and a future split may not be filtered.)
#
# AutoDrive-R2 integrates four dimensions with equal weights: position, steering,
# velocity, and temporal smoothness. Steering and velocity are not predicted
# directly here, so they are derived from consecutive waypoints (dt = 0.5 s),
# which is what a trajectory-only model makes available.
#
# The per-component scales below are the *measured* medians of our SFT model's
# errors on nuScenes val (n=5869), so a component of typical size contributes
# about 1.0 before weighting and no single term silently dominates:
#     position 0.674 m | velocity 0.516 m/s | heading 0.027 rad
#     speed jerk 0.032 m/s | heading jerk 0.001 rad
# Jerk medians are tiny, so those two use a coarser scale taken from p90 —
# normalising by the median would make ordinary smoothness variation swamp
# everything else.
# ---------------------------------------------------------------------------

DT = 0.5
POS_SCALE = 0.674
VEL_SCALE = 0.516
HEAD_SCALE = 0.30      # median 0.027 is degenerate (most frames drive straight);
                       # p75=0.155 / p90=1.341, so 0.30 keeps turns informative
JERK_V_SCALE = 0.176   # p90
JERK_TH_SCALE = 0.123  # p90
ERR_CLIP = 4.0
"""Each normalised component is clipped here. GRPO normalises advantages within
the group, so an unclipped squared error lets one bad rollout dictate the whole
group's direction."""

PHYSICS_WEIGHTS = {"pos": 1.0, "vel": 0.5, "head": 0.5, "smooth": 0.3}
"""Position stays dominant because it is what the benchmark scores; the dynamics
terms are shaping. AutoDrive-R2 sets all four to 1.0, but their reward is the only
signal, whereas here position also has to outrank three derived quantities
computed from the same six numbers."""


def _kinematics(pts: list[tuple[float, float]]):
    """Per-step speed (m/s) and heading (rad) implied by a waypoint list.

    The ego is at (0,0) at t=0, so the first step's delta is the waypoint itself.
    Heading uses atan2(dx, dy) because +y is forward in the ego frame.
    """
    prev = (0.0, 0.0)
    speed, head = [], []
    for x, y in pts:
        dx, dy = x - prev[0], y - prev[1]
        speed.append(math.hypot(dx, dy) / DT)
        head.append(math.atan2(dx, dy))
        prev = (x, y)
    return speed, head


def _ang_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return abs(d)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def physics_trajectory_reward(completions, solution=None, gt_mask=None, **kwargs):
    """Four-component physics reward, masked to the observed horizon.

    `gt_mask` is an optional per-sample list of 6 booleans marking which future
    steps were actually observed. Absent, every step counts, which is correct for
    the current train split (fully observed); it matters only if a split that is
    not pre-filtered is ever used.
    """
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        pred = parse_waypoints(_completion_text(comp))
        gold = parse_waypoints(solution[i] if i < len(solution) else "")
        if len(gold) < N_WAYPOINTS or len(pred) < N_WAYPOINTS:
            out.append(0.0)
            continue
        pred, gold = pred[:N_WAYPOINTS], gold[:N_WAYPOINTS]

        if gt_mask is not None and i < len(gt_mask) and gt_mask[i] is not None:
            keep = [j for j, f in enumerate(gt_mask[i]) if f]
        else:
            keep = list(range(N_WAYPOINTS))
        if not keep:
            out.append(0.0)
            continue

        pos = _mean(math.hypot(pred[j][0] - gold[j][0], pred[j][1] - gold[j][1])
                    for j in keep) / POS_SCALE

        pv, pth = _kinematics(pred)
        gv, gth = _kinematics(gold)
        vel = _mean(abs(pv[j] - gv[j]) for j in keep) / VEL_SCALE
        head = _mean(_ang_diff(pth[j], gth[j]) for j in keep) / HEAD_SCALE

        # Smoothness is a property of the prediction alone, so it is measured over
        # the whole horizon rather than only the observed part.
        jerk_v = _mean(abs(pv[j] - pv[j - 1]) for j in range(1, len(pv))) / JERK_V_SCALE
        jerk_th = _mean(_ang_diff(pth[j], pth[j - 1]) for j in range(1, len(pth))) / JERK_TH_SCALE

        err = (PHYSICS_WEIGHTS["pos"] * min(pos, ERR_CLIP)
               + PHYSICS_WEIGHTS["vel"] * min(vel, ERR_CLIP)
               + PHYSICS_WEIGHTS["head"] * min(head, ERR_CLIP)
               + PHYSICS_WEIGHTS["smooth"] * (min(jerk_v, ERR_CLIP) + min(jerk_th, ERR_CLIP)) / 2)
        total_w = sum(PHYSICS_WEIGHTS.values())
        # Map the weighted error onto (0, 1]: a sample matching our SFT model's
        # typical error lands near exp(-1) and the gradient stays live either side.
        out.append(math.exp(-err / total_w))
    return out


def trajectory_format_reward(completions, **kwargs) -> list[float]:
    """Full credit for exactly six parseable waypoints inside the answer tags.

    Separate from `format_reward` because the trajectory task has a stricter
    contract: the answer must be a waypoint list of the right length, not just
    non-empty text.
    """
    out = []
    for comp in completions:
        raw = _raw_text(comp)
        n = len(parse_waypoints(extract_answer(raw)))
        if n == N_WAYPOINTS:
            out.append(1.0 if has_format(raw) else 0.7)
        elif n > 0:
            out.append(0.3)
        else:
            out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# DriveLMM-o1. 이 데이터의 정답은 '**Step-by-Step Reasoning**: ... **Final Answer**: X'
# 형식이라 <think>/<answer> 를 읽는 기존 보상들이 그대로는 맞지 않는다.
# 공식 채점기(evaluation_script.py)와 같은 구분자 목록으로 최종 답변을 잘라 쓴다.
# ---------------------------------------------------------------------------

DLMM_MAX_TOKENS = 300
"""출력 길이 상한(토큰). 이보다 길면 벌점 — 위 dlmm_format_reward 주석 참조."""

DLMM_SPLITTERS = ("The final answer is:", "**Final Answer:**", "Final Answer", "Answer",
                  "Why take this action?:", "**Final Answer**", "**Final Decision**:",
                  "Final Step:", "<CONCLUSION>")


def _dlmm_final(text: str) -> str:
    for o in DLMM_SPLITTERS:
        if o in text:
            return text.split(o)[-1]
    return ""          # 구분자가 없으면 공식 채점에서 자동 오답 -> 여기서도 0 점


def dlmm_mcq_reward(completions, solution=None, **kwargs) -> list[float]:
    """선택지 문자 일치. 공식 MCQ 채점과 같은 규칙이다."""
    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        gold = OPTION_RE.findall((solution[i] if i < len(solution) else "").upper())
        pred = OPTION_RE.findall(_dlmm_final(_raw_text(comp)).upper())
        if not gold:
            out.append(0.0)          # 서술형 문항은 이 항이 0, text 보상이 담당
        else:
            out.append(1.0 if (pred and pred[0] == gold[0]) else 0.0)
    return out


def dlmm_text_reward(completions, solution=None, **kwargs) -> list[float]:
    """최종 답변과 정답의 단어 겹침. 서술형 문항(전체의 49%)의 유일한 내용 신호다.

    F1 대신 precision 을 2 배 가중한 F-beta(beta=0.5) 를 쓴다. 순수 F1 은 recall 로도
    오르기 때문에 길게 써서 정답 단어를 우연히 덮는 전략에 보상이 붙는다 — 실제로
    학습 중 이 항이 0.551 -> 0.572 로 오르는 동안 출력이 2 배 길어졌다.
    """
    solution = solution or []
    out = []
    for i, c in enumerate(completions):
        pred = _dlmm_final(_raw_text(c))
        gold = solution[i] if i < len(solution) else ""
        p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
        ov = sum((p & g).values())
        if not ov:
            out.append(0.0)
            continue
        pr, rc = ov / sum(p.values()), ov / sum(g.values())
        b2 = 0.25                       # beta=0.5 -> precision 을 4 배 무게
        out.append((1 + b2) * pr * rc / (b2 * pr + rc))
    return out


def dlmm_format_reward(completions, **kwargs) -> list[float]:
    """형식 + 길이 통제.

    측정으로 확인한 실패 모드: 상한을 600 토큰(= max_completion_length 640 과 거의 같음)
    으로 두었더니 출력이 574자 -> 1130자로 늘고 20.9% 가 '**Final Answer**' 에 닿기 전에
    잘렸다. 잘리면 공식 채점이 정답도 오답 처리하므로 MCQ 가 63.95% -> 51.90% 로 떨어졌다.
    그래서 (a) 상한을 300 토큰으로 낮추고, (b) 잘림에 음의 보상을 준다.
    SFT 출력이 평균 574자(~150 토큰)이므로 300 은 충분한 여유다.
    """
    out = []
    for comp in completions:
        raw = _raw_text(comp)
        n_raw = len(_tokens(raw))
        fin = _dlmm_final(raw)
        if "**Final Answer**" not in raw or not _tokens(fin):
            out.append(-1.0)          # 잘림/형식 실패는 0 이 아니라 벌점
        elif n_raw > DLMM_MAX_TOKENS:
            out.append(-0.5)          # 형식은 맞췄지만 장황
        else:
            out.append(1.0)
    return out





# ---------------------------------------------------------------- NuInstruct

def _nuins_task(kwargs, i: int) -> str:
    """GRPO 가 데이터셋 컬럼을 kwargs 로 넘겨준다. task 컬럼을 꺼낸다."""
    t = kwargs.get("task")
    if isinstance(t, (list, tuple)) and i < len(t):
        return t[i] or ""
    return t or ""


def nuins_task_reward(completions, solution=None, **kwargs) -> list[float]:
    """NuInstruct 공식 지표를 0~1 보상으로 환산한다.

    task 마다 지표가 달라서(MAE/Accuracy/MAP/BLEU) 하나의 척도로 묶어야 GRPO 의
    그룹 내 비교가 성립한다. 논문 Table 2 의 배정을 그대로 따르되:
      Accuracy 계열 -> 맞으면 1, 틀리면 0
      MAP 계열      -> 문항 AP 를 그대로 (이미 0~1)
      BLEU 계열     -> 문장 BLEU-4 (0~1)
      MAE 계열      -> 낮을수록 좋으므로 exp(-err/scale) 로 뒤집는다. scale 은
                       task 별 대표 오차 규모로 잡아 1 근처에서 포화하지 않게 한다.
    """
    from tools.nuinstruct_eval import (ACC_TASKS, BLEU_TASKS, MAE_TASKS, acc_score,
                                       ap_at_iou, mae_score, parse_objs)

    solution = solution or []
    # MAE 를 보상으로 바꿀 때 쓰는 스케일. 이 값 근처의 오차가 보상 0.37 이 된다.
    MAE_SCALE = {"perception-distance": 10.0, "perception-speed": 3.0,
                 "perception-instance_count": 1.0,
                 "prediction-motion_ego": 3.0, "prediction-motion_other": 5.0}
    out = []
    for i, comp in enumerate(completions):
        pred = _raw_text(comp)
        gold = solution[i] if i < len(solution) else ""
        task = _nuins_task(kwargs, i)
        if task in ACC_TASKS:
            out.append(1.0 if acc_score(pred, gold, task) else 0.0)
        elif task in MAE_TASKS:
            v = mae_score(pred, gold, task)
            if v is None or isinstance(v, tuple):
                out.append(0.0)                      # 파싱 실패
            else:
                out.append(float(math.exp(-v / MAE_SCALE.get(task, 5.0))))
        elif task in BLEU_TASKS:
            out.append(_f1(pred, gold))              # 문장 단위 BLEU 대용 (토큰 F1)
        else:                                        # risk-*
            out.append(float(ap_at_iou(parse_objs(pred), parse_objs(gold), 0.5)))
    return out


NUINS_FORMAT_TASKS = {"perception-closest", "perception-in_the_same_road",
                      "risk-approaching", "risk-braking", "risk-crossing",
                      "risk-lane_change", "risk-non", "risk-on_coming",
                      "risk-overtaking"}


def nuins_format_reward(completions, solution=None, **kwargs) -> list[float]:
    """요구 형식을 지켰는지. 객체를 나열해야 하는 문항에서 <class>[cN,...] 이
    안 나오면 지표가 통째로 0 이 되므로 별도 신호로 둔다."""
    from tools.nuinstruct_eval import parse_objs

    solution = solution or []
    out = []
    for i, comp in enumerate(completions):
        pred = _raw_text(comp)
        gold = solution[i] if i < len(solution) else ""
        task = _nuins_task(kwargs, i)
        if task not in NUINS_FORMAT_TASKS:
            out.append(1.0)
            continue
        want = bool(parse_objs(gold))
        has = bool(parse_objs(pred))
        if want:
            out.append(1.0 if has else -1.0)
        else:
            # 정답이 "없음" 인데 객체를 지어내면 벌점
            out.append(-1.0 if has else 1.0)
    return out




# ---------------------------------------------- NuInstruct: 연속화한 보상

NUINS_STATUS_TASKS = {"perception-status", "prediction-status_ego",
                      "prediction-status_others"}


def _soft_ap(preds, gts, thr: float = 0.5, below: float = 0.4) -> float:
    """AP 를 IoU 임계 아래에서도 부분 점수로 잇는다.

    원래 ap_at_iou 는 IoU 0.5 를 하드 컷오프로 써서, 카메라와 클래스는 맞췄는데
    박스가 조금 어긋난 롤아웃이 완전 오답과 같은 0 점을 받는다. 롤아웃 8 개가 모두
    같은 쪽에 떨어지면 GRPO 의 그룹 내 분산이 0 이 되어 기울기가 사라진다
    (실측: 학습 중 분산 0 인 그룹이 37.7%).
    """
    from tools.nuinstruct_eval import iou

    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    used = [False] * len(gts)
    scores = []
    for cls, cam, box in preds:
        best, bi = 0.0, -1
        for i, (gc, gcam, gbox) in enumerate(gts):
            if used[i] or gc != cls or gcam != cam:
                continue
            v = iou(box, gbox)
            if v > best:
                best, bi = v, i
        if bi < 0:
            scores.append(0.0)                    # 클래스/카메라부터 틀림
        elif best >= thr:
            used[bi] = True
            scores.append(1.0)
        else:
            used[bi] = True
            scores.append(below * best / thr)     # 부분 점수
    cum, prec, rec = 0.0, [], []
    for i, sc in enumerate(scores):
        cum += sc
        prec.append(cum / (i + 1))
        rec.append(min(cum / len(gts), 1.0))
    ap, prev = 0.0, 0.0
    for p_, r_ in zip(prec, rec):
        ap += p_ * (r_ - prev)
        prev = r_
    return ap


def nuins_task_soft_reward(completions, solution=None, **kwargs) -> list[float]:
    """nuins_task_reward 의 연속화 판. 평가 지표는 그대로 두고 학습 신호만 부드럽게 한다.

    이진 보상 task 가 전체의 53%(8,022/15,046)이고 그 포화도가 95~100% 다. 같은
    프롬프트의 롤아웃이 다 같은 쪽에 떨어져 학습이 안 되는 구간이 크다. 아래 세
    곳을 부분 점수로 바꾼다. 완전 정답이 항상 우세하도록 가중치를 잡아, 모델이
    어중간하게 답하는 쪽으로 유도되지 않게 한다.

      perception-closest : 카메라 0.5 + 클래스 0.3 + 0.2*IoU   (완전정답 1.0)
      status 계열        : 완전일치 1.0, 아니면 0.6*토큰F1
      risk-*             : _soft_ap (IoU 임계 아래 부분 점수)
    나머지(MAE/BLEU/yes-no)는 원래대로 둔다 — MAE 는 이미 연속이고 yes/no 는
    본질적으로 이진이라 손댈 수 없다.
    """
    from tools.nuinstruct_eval import (ACC_TASKS, BLEU_TASKS, MAE_TASKS, acc_score,
                                       iou, mae_score, parse_objs)

    solution = solution or []
    MAE_SCALE = {"perception-distance": 10.0, "perception-speed": 3.0,
                 "perception-instance_count": 1.0,
                 "prediction-motion_ego": 3.0, "prediction-motion_other": 5.0}
    out = []
    for i, comp in enumerate(completions):
        pred = _raw_text(comp)
        gold = solution[i] if i < len(solution) else ""
        task = _nuins_task(kwargs, i)

        if task == "perception-closest":
            go, po = parse_objs(gold), parse_objs(pred)
            if not go or not po:
                out.append(0.0)
                continue
            gc, gcam, gbox = go[0]
            pc, pcam, pbox = po[0]
            r = 0.0
            if pcam == gcam:
                r += 0.5
            if pc == gc:
                r += 0.3
            r += 0.2 * iou(pbox, gbox)
            out.append(min(r, 1.0))

        elif task in NUINS_STATUS_TASKS:
            out.append(1.0 if acc_score(pred, gold, task) else 0.6 * _f1(pred, gold))

        elif task in ACC_TASKS:                   # in_the_same_road (yes/no)
            out.append(1.0 if acc_score(pred, gold, task) else 0.0)

        elif task in MAE_TASKS:
            v = mae_score(pred, gold, task)
            out.append(0.0 if (v is None or isinstance(v, tuple))
                       else float(math.exp(-v / MAE_SCALE.get(task, 5.0))))

        elif task in BLEU_TASKS:
            out.append(_f1(pred, gold))

        else:                                     # risk-*
            out.append(float(_soft_ap(parse_objs(pred), parse_objs(gold))))
    return out


# ---------------------------------------------------------------- OmniDrive

# 공식 counterfactual 평가가 세는 키워드 (OmniDrive eval 스크립트와 동일).
OMNI_SAFETY_KW = (
    "unsafe", "safe", "run the red light", "collision",
    "out of the drivable area",
)
# 정답이 인용하는 자기중심 좌표. (+x 전방, +y 좌측) 이고 부호가 항상 붙는다.
OMNI_COORD_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")


def _omni_coords(text: str) -> list[tuple[float, float]]:
    out = []
    for a, b in OMNI_COORD_RE.findall(text or ""):
        try:
            out.append((float(a), float(b)))
        except ValueError:
            continue
    return out


def _omni_coord_f1(pred: str, gold: str, tol: float = 2.0) -> float:
    """정답이 인용한 객체 좌표를 얼마나 맞췄나 (L1 <= tol m 이면 일치).

    이 항이 없으면 언어 겹침만 남아 모델이 뷰를 볼 이유가 없다. OmniDrive 정답은
    객체 위치를 (x, y) 로 인용하고 그 위치는 특정 카메라에만 보이므로, 관점 접지가
    보상에 반영되는 경로는 사실상 이것뿐이다.
    """
    g = _omni_coords(gold)
    if not g:
        return -1.0                      # 좌표가 없는 문항 -> 이 항을 건너뛴다
    p = _omni_coords(pred)
    if not p:
        return 0.0
    used, hit = set(), 0
    for gx, gy in g:
        best, bi = None, -1
        for i, (px, py) in enumerate(p):
            if i in used:
                continue
            d = abs(px - gx) + abs(py - gy)
            if best is None or d < best:
                best, bi = d, i
        if bi >= 0 and best is not None and best <= tol:
            used.add(bi)
            hit += 1
    precision, recall = hit / len(p), hit / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _omni_kw_f1(pred: str, gold: str) -> float:
    g = _extract_phrases(gold, OMNI_SAFETY_KW)
    if not g:
        return -1.0
    p = _extract_phrases(pred, OMNI_SAFETY_KW)
    if not p:
        return 0.0
    n = len(p & g)
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def omni_task_reward(completions, solution=None, **kwargs) -> list[float]:
    """OmniDrive vqa/counterfactual 보상. 공식 지표 세 갈래를 0~1 로 묶는다.

    공식 평가는 CIDEr/BLEU/ROUGE(언어) + counterfactual 안전성 P/R(키워드)이다.
    거기에 좌표 항을 더한다 — 언어 겹침만 쓰면 관점 접지가 보상에 안 실린다.

    가중치는 항이 실제로 존재하는 문항에서만 재정규화한다. counterfactual 은
    97.2% 가 안전성 키워드를 담고, vqa 는 대부분 좌표를 담는다.
    """
    out = []
    for i, c in enumerate(completions):
        pred = _completion_text(c)
        gold = solution[i] if isinstance(solution, (list, tuple)) else (solution or "")
        parts = [(0.5, _f1(pred, gold))]
        kw = _omni_kw_f1(pred, gold)
        if kw >= 0.0:
            parts.append((0.25, kw))
        co = _omni_coord_f1(pred, gold)
        if co >= 0.0:
            parts.append((0.35, co))
        wsum = sum(w for w, _ in parts)
        out.append(sum(w * v for w, v in parts) / wsum)
    return out


def omni_format_reward(completions, solution=None, **kwargs) -> list[float]:
    """길이·반복만 본다. OmniDrive 정답은 자유 서술이라 강제할 형식이 없다.

    학습 전 모델을 바로 GRPO 로 돌리면 (a) 한 단어로 끝내거나 (b) 같은 문장을
    무한 반복하는 두 붕괴가 흔하다. 둘 다 언어 F1 로는 충분히 벌점이 안 된다.
    """
    out = []
    for i, c in enumerate(completions):
        pred = _completion_text(c)
        toks = _tokens(pred)
        gold = solution[i] if isinstance(solution, (list, tuple)) else (solution or "")
        n_gold = max(len(_tokens(gold)), 1)
        if not toks:
            out.append(0.0)
            continue
        # 정답 길이의 0.4~2.5배 안이면 1.0, 벗어나면 선형 감쇠
        ratio = len(toks) / n_gold
        if ratio < 0.4:
            length = ratio / 0.4
        elif ratio > 2.5:
            length = max(0.0, 1.0 - (ratio - 2.5) / 2.5)
        else:
            length = 1.0
        # 반복도: 서로 다른 3-gram 비율
        grams = [tuple(toks[j:j + 3]) for j in range(max(len(toks) - 2, 1))]
        variety = len(set(grams)) / max(len(grams), 1)
        out.append(0.5 * length + 0.5 * variety)
    return out


REWARD_REGISTRY = {
    "behavior": behavior_reward,
    "text_overlap": text_overlap_reward,
    "object_reference": object_reference_reward,
    "multiple_choice": multiple_choice_reward,
    "format": format_reward,
    "trajectory_l2": trajectory_l2_reward,
    "physics_trajectory": physics_trajectory_reward,
    "trajectory_format": trajectory_format_reward,
    "dlmm_mcq": dlmm_mcq_reward,
    "dlmm_text": dlmm_text_reward,
    "dlmm_format": dlmm_format_reward,
    "nuins_task": nuins_task_reward,
    "nuins_format": nuins_format_reward,
    "nuins_task_soft": nuins_task_soft_reward,
    "omni_task": omni_task_reward,
    "omni_format": omni_format_reward,
}


def build_rewards(names: list[str]):
    missing = [n for n in names if n not in REWARD_REGISTRY]
    if missing:
        raise ValueError(f"unknown reward(s): {missing}; available: {sorted(REWARD_REGISTRY)}")
    return [REWARD_REGISTRY[n] for n in names]
