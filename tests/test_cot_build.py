"""Tests for the CoT label-construction pieces: format contract, trajectory-derived
behaviour, and the teacher-output verification that gates pass 1.

CPU only, no teacher endpoint needed.

    python -m pytest tests/test_cot_build.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))

from behavior import (  # noqa: E402
    behavior_from_text,
    behavior_from_waypoints,
    agrees_with,
    parse_waypoints,
)
from cot_tasks import DriveLMTask  # noqa: E402

from vgrl.prompt_format import (  # noqa: E402
    FORMAT_INSTRUCTION,
    SYSTEM_DRIVELM,
    SYSTEM_DRIVELM_PLAIN,
    SYSTEM_NUSCENES,
    SYSTEM_NUSCENES_PLAIN,
    extract_answer,
    extract_think,
    has_format,
    mentions_camera,
    strip_camera_names,
    wrap,
)


# ------------------------------------------------------------- format contract

def test_wrap_roundtrip():
    body = wrap("first I look, then I decide", "[(0.10,1.71)]")
    assert has_format(body)
    assert extract_think(body) == "first I look, then I decide"
    assert extract_answer(body) == "[(0.10,1.71)]"


def test_extract_answer_falls_back_to_tail_after_think():
    text = "<think>reasoned about it</think>\nThe answer is B."
    assert not has_format(text)
    assert extract_answer(text) == "The answer is B."


def test_extract_answer_falls_back_to_whole_string():
    assert extract_answer("just an answer") == "just an answer"


def test_extract_answer_strict_mode_returns_empty():
    assert extract_answer("no tags here", fallback_to_tail=False) == ""


def test_plain_systems_do_not_request_the_format():
    for cot, plain in ((SYSTEM_DRIVELM, SYSTEM_DRIVELM_PLAIN),
                       (SYSTEM_NUSCENES, SYSTEM_NUSCENES_PLAIN)):
        assert FORMAT_INSTRUCTION in cot
        assert FORMAT_INSTRUCTION not in plain


def test_no_system_prompt_asks_for_view_attribution():
    """The whole point of the method: the CoT must not be trained to announce which
    view its evidence came from, or the RL term measures a phrase, not perception."""
    for s in (SYSTEM_DRIVELM, SYSTEM_NUSCENES, SYSTEM_DRIVELM_PLAIN, SYSTEM_NUSCENES_PLAIN):
        low = s.lower()
        for phrase in ("which view", "which camera", "state the view", "answer view",
                       "evidence view", "name the camera"):
            assert phrase not in low, f"{phrase!r} leaked into a system prompt"


def test_camera_helpers():
    t = "The bus in CAM_BACK_LEFT is stationary."
    assert mentions_camera(t)
    assert not mentions_camera("The bus behind is stationary.")
    assert "CAM_BACK_LEFT" not in strip_camera_names(t)


# ------------------------------------------------- behaviour from a trajectory

def test_parse_waypoints_handles_negatives_and_spaces():
    assert parse_waypoints("[(-0.11,-11.70), (0.5, 3.6)]") == [(-0.11, -11.70), (0.5, 3.6)]


def test_straight_and_constant():
    wps = [(0.0, 5.0), (0.0, 10.0), (0.0, 15.0), (0.0, 20.0), (0.0, 25.0), (0.0, 30.0)]
    b = behavior_from_waypoints(wps)
    assert b.lateral == "straight" and b.longitudinal == "constant"


def test_right_turn_detected():
    wps = [(0.2, 2.0), (0.8, 4.0), (2.0, 5.5), (4.0, 6.5), (6.5, 7.0), (9.0, 7.2)]
    assert behavior_from_waypoints(wps).lateral == "right"


def test_left_turn_detected():
    wps = [(-0.2, 2.0), (-0.8, 4.0), (-2.0, 5.5), (-4.0, 6.5), (-6.5, 7.0), (-9.0, 7.2)]
    assert behavior_from_waypoints(wps).lateral == "left"


def test_deceleration_to_stop():
    wps = [(0.0, 4.0), (0.0, 7.0), (0.0, 9.0), (0.0, 10.0), (0.0, 10.4), (0.0, 10.5)]
    b = behavior_from_waypoints(wps)
    assert b.longitudinal == "stop"


def test_stationary_when_nothing_moves():
    wps = [(0.0, 0.0)] * 6
    assert behavior_from_waypoints(wps).longitudinal == "stationary"


def test_acceleration():
    wps = [(0.0, 1.0), (0.0, 3.0), (0.0, 6.0), (0.0, 10.0), (0.0, 15.0), (0.0, 21.0)]
    assert behavior_from_waypoints(wps).longitudinal in {"accelerate", "rapid_accelerate"}


def test_too_short_is_ungradable():
    assert behavior_from_waypoints([(0.0, 1.0)]) is None


# -------------------------------------------------- behaviour claimed in text

def test_behavior_from_text_longest_phrase_wins():
    lat, lon = behavior_from_text(
        "I will maintain current lane and apply rapid deceleration to avoid the truck."
    )
    assert lat == "straight" and lon == "rapid_decelerate"


def test_behavior_from_text_none_when_silent():
    assert behavior_from_text("The scene is bright and clear.") == (None, None)


def test_agreement_accepts_compatible_longitudinals():
    truth = behavior_from_waypoints(
        [(0.0, 4.0), (0.0, 7.0), (0.0, 9.0), (0.0, 10.0), (0.0, 10.4), (0.0, 10.5)]
    )
    assert truth.longitudinal == "stop"
    # "decelerate" for a trajectory that ends stopped is not a contradiction
    assert agrees_with(truth, "straight", "decelerate")
    assert not agrees_with(truth, "straight", "accelerate")


def test_agreement_rejects_wrong_lateral():
    """A perfectly straight path is outside the ambiguity band, so naming a turn
    direction is a real contradiction, not a boundary disagreement."""
    truth = behavior_from_waypoints([(0.0, 5.0)] + [(0.0, 5.0 * i) for i in range(2, 7)])
    assert abs(truth.lateral_offset) == 0.0
    assert not agrees_with(truth, "left", "constant")
    assert not agrees_with(truth, "right", "constant")


def test_unstated_axis_is_permissive_by_default_and_strict_on_request():
    truth = behavior_from_waypoints([(0.0, 5.0 * i) for i in range(1, 7)])
    assert agrees_with(truth, "straight", None)
    assert not agrees_with(truth, "straight", None, require_both=True)


# ---------------------------------------------- DriveLM teacher-output parsing

def _drivelm_task():
    task = DriveLMTask.__new__(DriveLMTask)  # no file I/O needed for these
    task.f1_threshold = 0.45
    return task


def test_split_reasoning_answer():
    task = _drivelm_task()
    r, a = task.split_reasoning_answer(
        "REASONING: the sedan ahead is braking, so closing speed matters.\n"
        "ANSWER: It is decelerating."
    )
    assert r.startswith("the sedan ahead")
    assert a == "It is decelerating."


def test_split_without_headings_is_all_reasoning():
    task = _drivelm_task()
    r, a = task.split_reasoning_answer("just some free text")
    assert r == "just some free text" and a == ""


def test_verify_rejects_missing_answer():
    task = _drivelm_task()
    sample = {"context": "What is it doing?", "gt_answer": "It is stopped.",
              "meta": {"category": "perception"}}
    ok, reason = task.verify("REASONING: " + "word " * 40, sample)
    assert not ok and reason == "no_answer_stated"


def test_verify_rejects_short_reasoning():
    task = _drivelm_task()
    sample = {"context": "What is it doing?", "gt_answer": "It is stopped.",
              "meta": {"category": "perception"}}
    ok, reason = task.verify("REASONING: brief.\nANSWER: It is stopped.", sample)
    assert not ok and reason == "too_short"


def test_verify_accepts_matching_answer():
    task = _drivelm_task()
    gt = "There is a brown SUV to the back of the ego vehicle, <c1,CAM_BACK,1088.3,497.5>."
    sample = {"context": "What are the important objects?", "gt_answer": gt,
              "meta": {"category": "perception"}}
    text = ("REASONING: " + "looking at the rear view a large brown vehicle sits behind "
            "the ego car and nothing else is close. " * 2 + "\nANSWER: " + gt)
    ok, reason = task.verify(text, sample)
    assert ok, reason


def test_verify_rejects_contradicting_answer():
    task = _drivelm_task()
    sample = {
        "context": "What is the moving status of the ego vehicle?",
        "gt_answer": "The ego vehicle is going straight. The speed is slow.",
        "meta": {"category": "behavior"},
    }
    text = ("REASONING: " + "the road curves and the vehicle follows it around. " * 4 +
            "\nANSWER: The ego vehicle is turning left. The speed is fast.")
    ok, reason = task.verify(text, sample)
    assert not ok and reason == "behavior_mismatch"


def test_verify_behavior_exact_match():
    task = _drivelm_task()
    gt = "The ego vehicle is going straight. The speed is slow."
    sample = {"context": "moving status?", "gt_answer": gt, "meta": {"category": "behavior"}}
    text = ("REASONING: " + "lane markings run parallel and the scene barely shifts "
            "between frames so speed is low. " * 2 + "\nANSWER: " + gt)
    ok, reason = task.verify(text, sample)
    assert ok, reason


def test_verify_multiple_choice_option():
    task = _drivelm_task()
    sample = {
        "context": "Please select the best answer. A. brake B. accelerate C. hold D. swerve",
        "gt_answer": "C",
        "meta": {"category": "planning"},
    }
    good = ("REASONING: " + "traffic ahead is steady and there is no hazard requiring a "
            "change of speed at this moment. " * 2 + "\nANSWER: C")
    bad = good.replace("ANSWER: C", "ANSWER: A")
    assert task.verify(good, sample)[0]
    assert not task.verify(bad, sample)[0]


# --------------------------------------------------------- nuScenes verification

def test_nuscenes_verify_rejects_leaked_coordinates():
    from cot_tasks import NuScenesPlanningTask

    task = NuScenesPlanningTask.__new__(NuScenesPlanningTask)
    gt = "[(0.00,5.00), (0.00,10.00), (0.00,15.00), (0.00,20.00), (0.00,25.00), (0.00,30.00)]"
    sample = {"gt_answer": gt, "meta": {}}
    text = ("<think>" + "the road is clear so I will go straight and maintain current "
            "speed. " * 3 + "the plan is [(0.00,5.00), (0.00,10.00)]</think>")
    ok, reason = task.verify(text, sample)
    assert not ok and reason == "leaked_coordinates"


def test_nuscenes_verify_rejects_contradiction():
    from cot_tasks import NuScenesPlanningTask

    task = NuScenesPlanningTask.__new__(NuScenesPlanningTask)
    gt = "[(0.00,5.00), (0.00,10.00), (0.00,15.00), (0.00,20.00), (0.00,25.00), (0.00,30.00)]"
    sample = {"gt_answer": gt, "meta": {}}
    text = ("<think>" + "a pedestrian steps out so I must turn left and apply emergency "
            "brake immediately to stay safe. " * 3 + "</think>")
    ok, reason = task.verify(text, sample)
    assert not ok and reason.startswith("contradicts_trajectory")


def test_nuscenes_verify_accepts_consistent_reasoning():
    from cot_tasks import NuScenesPlanningTask

    task = NuScenesPlanningTask.__new__(NuScenesPlanningTask)
    gt = "[(0.00,5.00), (0.00,10.00), (0.00,15.00), (0.00,20.00), (0.00,25.00), (0.00,30.00)]"
    sample = {"gt_answer": gt, "meta": {}}
    text = ("<think>" + "the lane ahead is clear and the light is green, so I will go "
            "straight and maintain current speed. " * 3 + "</think>")
    ok, reason = task.verify(text, sample)
    assert ok, reason


def test_nuscenes_verify_requires_a_decision():
    from cot_tasks import NuScenesPlanningTask

    task = NuScenesPlanningTask.__new__(NuScenesPlanningTask)
    gt = "[(0.00,5.00), (0.00,10.00), (0.00,15.00), (0.00,20.00), (0.00,25.00), (0.00,30.00)]"
    sample = {"gt_answer": gt, "meta": {}}
    text = "<think>" + "the weather is clear and buildings line the street. " * 6 + "</think>"
    ok, reason = task.verify(text, sample)
    assert not ok and reason == "no_decision_stated"


# ------------------------------------------------------------ reward alignment

def test_rewards_score_only_the_answer_span():
    """A generation that pads its reasoning with the reference wording must not get
    content credit for it."""
    from vgrl.rewards import text_overlap_reward

    gt = "the brown SUV behind the ego vehicle is stationary"
    cheating = wrap(gt + " " + gt, "unrelated words entirely")
    honest = wrap("I checked the rear view.", gt)
    scores = text_overlap_reward([cheating, honest], solution=[gt, gt])
    assert scores[1] > scores[0]


def test_format_reward_tiers():
    from vgrl.rewards import format_reward

    good = wrap("reasoned", "a fine answer")
    untagged = "a fine answer"
    empty = wrap("reasoned", "")
    runaway = wrap("reasoned", "word " * 400)
    assert format_reward([good, untagged, empty, runaway]) == [1.0, 0.5, 0.0, 0.0]


# ------------------------------------------- regressions from the pilot run
#
# A first pilot rejected ~40% of *correct* teacher output. Three separate bugs;
# each gets a test so they cannot come back.

def test_scene_mentioning_stopped_vehicles_is_not_a_stop_decision():
    """Bug 1: a bare "stop" keyword matched "stopped vehicles" in the scene
    description, turning 11 of 39 correct CoTs into 'contradicts_trajectory'."""
    text = (
        "Step 1: Scene. Several vehicles are stopped behind the ego vehicle and the "
        "light is green.\n"
        "Step 4: Decision. The correct manoeuvre is going straight while accelerating "
        "gently."
    )
    lat, lon = behavior_from_text(text)
    assert lat == "straight"
    assert lon == "accelerate", f"got {lon!r} — a scene 'stopped' leaked into the decision"


def test_prose_moving_forward_does_not_override_a_turn_decision():
    """Bug 2: "forward" was a trigger for straight, and 36 of 39 CoTs said
    "moving forward" somewhere in their reasoning."""
    text = (
        "Step 3: Reasoning. This aligns with the mission goal of moving forward.\n"
        "Step 4: Decision. The correct manoeuvre is to turn right while decelerating "
        "gently."
    )
    lat, _ = behavior_from_text(text)
    assert lat == "right", f"got {lat!r} — prose 'moving forward' beat the decision"


def test_hint_wording_is_matchable_by_the_checker():
    """Bug 3: hint_for said "turning/moving right", which was in neither phrase
    table, so a teacher echoing the supplied answer was rejected for it."""
    from cot_tasks import NuScenesPlanningTask

    for lat_truth, expect in (("straight", "straight"), ("left", "left"), ("right", "right")):
        for lon_truth in ("constant", "accelerate", "decelerate", "rapid_accelerate",
                          "rapid_decelerate", "stop", "stationary"):
            sample = {"meta": {"truth_lateral": lat_truth, "truth_longitudinal": lon_truth}}
            hint = NuScenesPlanningTask.hint_for(sample)
            lat, lon = behavior_from_text(f"Decision: the correct manoeuvre is {hint}.")
            assert lat == expect, f"hint {hint!r} -> lat {lat!r}, wanted {expect!r}"
            assert lon is not None, f"hint {hint!r} -> no longitudinal match"


def test_decision_span_prefers_the_decision_heading():
    from behavior import decision_span

    text = "Scene: vehicles are stopped.\n### Step 4: Decision\nGo straight and accelerate."
    span = decision_span(text)
    assert "Go straight" in span and "vehicles are stopped" not in span


def test_decision_span_falls_back_to_the_tail():
    from behavior import decision_span

    text = "One. Two. Three. Four. Five."
    assert "Five." in decision_span(text)


def test_small_lateral_offset_accepts_straight_or_the_drift_side():
    """Every lateral disagreement in the pilot had |x_end| < 2 m — lane-drift
    scale, where 'straight' is a reasonable description."""
    wps = [(0.2, 3.0), (0.5, 6.0), (0.8, 9.0), (1.1, 12.0), (1.4, 15.0), (1.6, 18.0)]
    truth = behavior_from_waypoints(wps)
    assert truth.lateral == "right" and abs(truth.lateral_offset) < 2.5
    assert agrees_with(truth, "straight", None)
    assert agrees_with(truth, "right", None)
    assert not agrees_with(truth, "left", None)


def test_large_lateral_offset_still_rejects_straight():
    wps = [(-1.0, 2.0), (-2.5, 4.0), (-4.5, 5.5), (-7.0, 6.5), (-9.5, 7.0), (-12.0, 7.2)]
    truth = behavior_from_waypoints(wps)
    assert not agrees_with(truth, "straight", None)


def test_small_speed_change_accepts_either_direction():
    # dv well under 1 m/s: constant / accelerate / decelerate are all defensible
    wps = [(0.0, 3.0), (0.0, 6.05), (0.0, 9.15), (0.0, 12.3), (0.0, 15.5), (0.0, 18.75)]
    truth = behavior_from_waypoints(wps)
    assert abs(truth.speed_end - truth.speed_start) < 1.0
    for said in ("constant", "accelerate", "decelerate"):
        assert agrees_with(truth, "straight", said), said


def test_clear_acceleration_still_rejects_stop_and_decelerate():
    wps = [(0.0, 1.5), (0.0, 3.4), (0.0, 5.6), (0.0, 8.1), (0.0, 10.9), (0.0, 14.0)]
    truth = behavior_from_waypoints(wps)
    assert truth.speed_end - truth.speed_start > 2.0
    for said in ("stop", "decelerate", "rapid_decelerate", "constant"):
        assert not agrees_with(truth, "straight", said), said
    assert agrees_with(truth, "straight", "accelerate")


def test_ambiguity_band_has_a_lower_edge():
    """Inside the band both readings pass; below it only 'straight' does."""
    from behavior import LATERAL_AMBIGUOUS_M, LATERAL_STRAIGHT_M

    def straight_with_offset(x_end):
        n = 6
        return [(x_end * (i + 1) / n, 5.0 * (i + 1)) for i in range(n)]

    clear = behavior_from_waypoints(straight_with_offset(0.2))
    assert abs(clear.lateral_offset) < LATERAL_STRAIGHT_M / 2
    assert agrees_with(clear, "straight", None)
    assert not agrees_with(clear, "right", None), "a dead-straight path must not accept a turn"

    banded = behavior_from_waypoints(straight_with_offset(1.5))
    assert LATERAL_STRAIGHT_M / 2 <= abs(banded.lateral_offset) < LATERAL_AMBIGUOUS_M
    assert agrees_with(banded, "straight", None) and agrees_with(banded, "right", None)
