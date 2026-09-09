"""Derive the driving behaviour that a ground-truth trajectory actually describes.

This is the automatic quality check for nuScenes CoT labels. MindDriver's pipeline
paid a second LLM call per sample to ask "is this reasoning correct?"
(gen_data/check.py) — a weak and expensive signal. The trajectory itself already
says what the ego did, so a teacher CoT that concludes "turn left, accelerate" for
a trajectory that goes straight and brakes can be rejected for free and
deterministically.

Ego frame follows MindDriver's convention: X is lateral (positive right), Y is
forward, waypoints are cumulative displacements at 0.5 s intervals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DT = 0.5

# thresholds, in metres / (m/s) — chosen so that the label matches what a human
# would call the manoeuvre, and documented because the QC pass depends on them
LATERAL_STRAIGHT_M = 1.0     # |x| at 3 s below this is "straight"
LATERAL_TURN_M = 4.0         # above this it is a turn rather than a lane shift
SPEED_EPS = 0.5              # m/s change below this is "constant"
SPEED_FAST_EPS = 2.0         # above this it is "rapid"
STOPPED_SPEED = 0.5          # m/s

LATERAL_LABELS = ("straight", "left", "right")
LONGITUDINAL_LABELS = (
    "constant", "accelerate", "decelerate", "rapid_accelerate",
    "rapid_decelerate", "stop", "stationary",
)

PAIR_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")

# Phrases a teacher model is likely to use, mapped onto our labels. Matched
# longest-first so "rapid deceleration" wins over "deceleration".
#
# Two rules learned the hard way from a pilot run, where a buggy version of this
# table rejected ~40% of *correct* teacher output:
#
#   - no bare "stop": 11 of 39 CoTs said "stopped vehicles" while describing the
#     scene, which silently became a longitudinal decision of "stop";
#   - no bare "forward": 36 of 39 said "moving forward" somewhere, which made
#     almost every sample look like it had decided to go straight.
#
# The structural guard is `decision_span()` below — matching only the decision
# section rather than the whole CoT. These tables stay conservative anyway, since
# a missed phrase costs one sample while a false match corrupts a label.
LATERAL_PHRASES = {
    "straight": ("go straight", "going straight", "continue straight", "straight ahead",
                 "maintain current lane", "keep lane", "stay in lane", "maintain lane",
                 "keep going straight", "move forward", "moving forward",
                 "continue forward", "proceed forward"),
    "left": ("turn left", "turning left", "turn/move left", "turning/moving left",
             "change lane left", "lane change to the left", "move to the left",
             "moving left", "left turn", "bear left", "veer left"),
    "right": ("turn right", "turning right", "turn/move right", "turning/moving right",
              "change lane right", "lane change to the right", "move to the right",
              "moving right", "right turn", "bear right", "veer right"),
}
LONGITUDINAL_PHRASES = {
    "rapid_decelerate": ("emergency brake", "emergency braking", "rapid deceleration",
                         "brake hard", "hard braking", "decelerate rapidly",
                         "decelerating hard", "braking hard"),
    "rapid_accelerate": ("rapid acceleration", "accelerate rapidly", "speed up quickly",
                         "accelerating hard", "accelerate hard"),
    "stop": ("come to a stop", "coming to a stop", "slow to a stop", "slowing to a stop",
             "decelerate to zero", "stop the vehicle", "bring the vehicle to a stop"),
    "stationary": ("remain stationary", "remaining stationary", "stay stopped",
                   "remain still", "keep stopped", "staying stationary"),
    "decelerate": ("smooth deceleration", "decelerate gently", "decelerating gently",
                   "decelerate", "decelerating", "slow down", "slowing down",
                   "reduce speed", "reducing speed"),
    "accelerate": ("smooth acceleration", "accelerate gently", "accelerating gently",
                   "accelerate", "accelerating", "speed up", "increase speed",
                   "increasing speed"),
    "constant": ("maintain current speed", "maintain speed", "maintaining speed",
                 "constant speed", "keep the same speed", "steady speed",
                 "holding a constant speed"),
}

# Where the teacher states its conclusion. The CoT template asks for a numbered
# "Decision" step, and models render that heading in many ways.
DECISION_MARKERS = (
    "decision", "결정", "final answer", "conclusion", "summary of reasoning",
    "action decision",
)


@dataclass
class Behavior:
    lateral: str
    longitudinal: str
    lateral_offset: float
    speed_start: float
    speed_end: float

    def describe(self) -> str:
        return f"{self.lateral} / {self.longitudinal}"


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in PAIR_RE.findall(text or "")]


def behavior_from_waypoints(
    waypoints: list[tuple[float, float]],
    speed_before: float | None = None,
) -> Behavior | None:
    """Classify a 6-waypoint plan. Returns None if it is too short to judge."""
    if len(waypoints) < 2:
        return None

    xs = [p[0] for p in waypoints]
    step_speeds = []
    prev = (0.0, 0.0)
    for p in waypoints:
        step_speeds.append(math.dist(p, prev) / DT)
        prev = p

    v_start = speed_before if speed_before is not None else step_speeds[0]
    v_end = step_speeds[-1]
    dv = v_end - v_start

    lateral_offset = xs[-1]
    if abs(lateral_offset) < LATERAL_STRAIGHT_M:
        lateral = "straight"
    else:
        lateral = "right" if lateral_offset > 0 else "left"

    if max(step_speeds) < STOPPED_SPEED and v_start < STOPPED_SPEED:
        longitudinal = "stationary"
    elif v_end < STOPPED_SPEED:
        longitudinal = "stop"
    elif dv > SPEED_FAST_EPS:
        longitudinal = "rapid_accelerate"
    elif dv < -SPEED_FAST_EPS:
        longitudinal = "rapid_decelerate"
    elif dv > SPEED_EPS:
        longitudinal = "accelerate"
    elif dv < -SPEED_EPS:
        longitudinal = "decelerate"
    else:
        longitudinal = "constant"

    return Behavior(lateral, longitudinal, lateral_offset, v_start, v_end)


def _match_phrases(text: str, table: dict[str, tuple[str, ...]]) -> str | None:
    low = re.sub(r"\s+", " ", (text or "").lower())
    best, best_len = None, -1
    for label, phrases in table.items():
        for ph in phrases:
            if ph in low and len(ph) > best_len:
                best, best_len = label, len(ph)
    return best


def decision_span(text: str, min_words: int = 4) -> str:
    """The part of a CoT that states the decision.

    Matching manoeuvre phrases against the whole CoT is unsafe: a scene
    description that mentions "stopped vehicles" or "moving forward" reads as a
    decision. The prompt asks for a numbered "Decision" step, so prefer the text
    after the last such heading; otherwise fall back to the tail, which is where a
    conclusion lives even when the model skipped the heading.
    """
    body = text or ""
    low = body.lower()
    best = -1
    for marker in DECISION_MARKERS:
        idx = low.rfind(marker)
        if idx > best:
            best = idx
    if best >= 0:
        tail = body[best:]
        if len(tail.split()) >= min_words:
            return tail
    # no usable heading: last few sentences
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    return " ".join(sentences[-3:]) if sentences else body


def behavior_from_text(text: str, whole_text: bool = False) -> tuple[str | None, str | None]:
    """What manoeuvre a piece of generated reasoning claims to take.

    Reads only the decision section by default; `whole_text=True` scans everything
    (useful for diagnostics, not for gating labels).
    """
    scope = text if whole_text else decision_span(text)
    return _match_phrases(scope, LATERAL_PHRASES), _match_phrases(scope, LONGITUDINAL_PHRASES)


# Longitudinal labels that should not be treated as contradictions of each other:
# a teacher saying "decelerate" for a trajectory that ends stopped is not wrong.
_LONGITUDINAL_COMPATIBLE = {
    "stop": {"stop", "decelerate", "rapid_decelerate", "stationary"},
    "stationary": {"stationary", "stop", "constant"},
    "decelerate": {"decelerate", "rapid_decelerate", "stop"},
    "rapid_decelerate": {"rapid_decelerate", "decelerate", "stop"},
    "accelerate": {"accelerate", "rapid_accelerate", "constant"},
    "rapid_accelerate": {"rapid_accelerate", "accelerate"},
    "constant": {"constant", "accelerate", "decelerate"},
}

# Tolerance bands. The derived manoeuvre label is a hard threshold on a continuous
# quantity, so right at the boundary it is genuinely ambiguous and the teacher
# disagreeing with it is not an error. A pilot run made this concrete: every one of
# 13 lateral "disagreements" had |x_end| < 2 m — lane-drift scale, which a human
# would also call going straight — and 11 of 25 longitudinal ones sat under
# 1 m/s of speed change.
#
# The bands only widen what counts as *compatible*; they do not soften real
# contradictions. A trajectory that gains 2.4 m/s while the teacher says "stop"
# still fails, which is what this gate exists to catch.
LATERAL_AMBIGUOUS_M = 2.5
SPEED_AMBIGUOUS = 1.0


def agrees_with(
    truth: Behavior,
    claimed_lateral: str | None,
    claimed_longitudinal: str | None,
    require_both: bool = False,
) -> bool:
    """Is generated reasoning consistent with what the trajectory does?

    Unstated axes count as consistent — the point is to catch contradictions, not
    to force the teacher into a fixed vocabulary. `require_both` makes an
    unstated axis a failure instead.
    """
    if claimed_lateral is None:
        lat_ok = not require_both
    else:
        allowed_lat = {truth.lateral}
        offset = abs(truth.lateral_offset)
        # Only *inside the ambiguous band* do "straight" and the drift side both
        # count. Below the lower edge the path is unambiguously straight, so naming
        # a turn direction is a real error — not a boundary disagreement.
        if LATERAL_STRAIGHT_M / 2 <= offset < LATERAL_AMBIGUOUS_M:
            allowed_lat |= {"straight", "right" if truth.lateral_offset > 0 else "left"}
        lat_ok = claimed_lateral in allowed_lat

    if claimed_longitudinal is None:
        lon_ok = not require_both
    else:
        allowed = set(_LONGITUDINAL_COMPATIBLE.get(truth.longitudinal, {truth.longitudinal}))
        dv = truth.speed_end - truth.speed_start
        if abs(dv) < SPEED_AMBIGUOUS and truth.longitudinal not in {"stop", "stationary"}:
            allowed |= {"constant", "accelerate", "decelerate"}
        lon_ok = claimed_longitudinal in allowed

    return lat_ok and lon_ok
