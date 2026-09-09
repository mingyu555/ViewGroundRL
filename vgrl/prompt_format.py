"""The one place the `<think>` / `<answer>` contract is defined.

SFT teaches this format, the RL stage asks for it, and the reward functions parse
it — so all three read the constants from here rather than each carrying their own
copy that can drift.

Deliberately absent: any instruction for the model to name which camera view the
answer came from. The reasoning is meant to be ordinary chain-of-thought. If the
CoT announced its evidence view, the model could learn to emit that phrase without
having looked, and the view-grounding RL term would be measuring a text pattern
instead of perception. Incidental mentions of a camera in prose are fine and are
not filtered by default; `--forbid_camera_mentions` in the CoT builder tightens
that if an experiment needs it.
"""

from __future__ import annotations

import re

from .views import CAMERAS

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

VIEW_HEADER = "".join(f"{cam}: <image>\n" for cam in CAMERAS)

FORMAT_INSTRUCTION = (
    f"Reason step by step inside {THINK_OPEN}...{THINK_CLOSE}, then give only the "
    f"final answer inside {ANSWER_OPEN}...{ANSWER_CLOSE}."
)

SYSTEM_DRIVELM = (
    "You are the perception and reasoning module of an autonomous vehicle. You are "
    "given the six surround-view camera images of the current frame, in this order: "
    + ", ".join(CAMERAS)
    + ". Objects are referred to as <cID,CAMERA,x,y>, where CAMERA names the view the "
    "object appears in and (x,y) is its pixel location in that view. Answer using the "
    "same reference format. " + FORMAT_INSTRUCTION
)

# The non-CoT variants, for the baseline SFT paths (data_prep/drivelm_prepare.py,
# data_prep/nuscenes_sft.py) whose targets are bare answers. Keeping them separate
# means no path ever asks for a format its labels do not contain.
SYSTEM_DRIVELM_PLAIN = SYSTEM_DRIVELM.replace(" " + FORMAT_INSTRUCTION, "")

SYSTEM_NUSCENES = (
    "You are the planning module of an autonomous vehicle. Coordinates: the X axis is "
    "perpendicular to and the Y axis parallel to your heading; you are at (0,0); units "
    "are metres. You are given the six surround-view camera images of the current "
    "frame, in this order: " + ", ".join(CAMERAS) + ". " + FORMAT_INSTRUCTION
)

NUSCENES_INSTRUCTION = (
    "Plan the ego vehicle's waypoints for the next 3 seconds at 0.5 s intervals and "
    "output them as [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]."
)

_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.S)
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.S)
_CAMERA_RE = re.compile(r"\b(?:" + "|".join(CAMERAS) + r")\b")


def wrap(cot: str, answer: str) -> str:
    """Build one assistant target."""
    return (
        f"{THINK_OPEN}\n{cot.strip()}\n{THINK_CLOSE}\n"
        f"{ANSWER_OPEN}{answer.strip()}{ANSWER_CLOSE}"
    )


def extract_think(text: str) -> str | None:
    m = _THINK_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_answer(text: str, fallback_to_tail: bool = True) -> str:
    """Contents of `<answer>`.

    With `fallback_to_tail`, a generation that reasoned but never opened an
    `<answer>` tag falls back to the text after `</think>` (or the whole string),
    so a missing tag degrades to "score the tail" rather than "score nothing".
    """
    m = _ANSWER_RE.search(text or "")
    if m:
        return m.group(1).strip()
    if not fallback_to_tail:
        return ""
    if THINK_CLOSE in (text or ""):
        return text.split(THINK_CLOSE, 1)[1].strip()
    return (text or "").strip()


def has_format(text: str) -> bool:
    return bool(_THINK_RE.search(text or "")) and bool(_ANSWER_RE.search(text or ""))


def mentions_camera(text: str) -> bool:
    return bool(_CAMERA_RE.search(text or ""))


def strip_camera_names(text: str, replacement: str = "one of the views") -> str:
    return _CAMERA_RE.sub(replacement, text or "")


SYSTEM_NUSCENES_PLAIN = SYSTEM_NUSCENES.replace(" " + FORMAT_INSTRUCTION, "")
