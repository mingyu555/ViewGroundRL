"""Camera-view bookkeeping for view-grounded RL.

The six nuScenes surround cameras are treated as an ordered list; every sample's
`images` field must present them in exactly this order, because the masking code
addresses a view by its positional index within the sample's image list.

DriveLM annotates referenced objects inline as `<c1,CAM_BACK,1088.3,497.5>`, so
the set of views that actually carry the evidence for a QA pair can be recovered
from the text itself — no extra labelling needed.
"""

from __future__ import annotations

import re

CAMERAS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAM_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CAMERAS)}
NUM_VIEWS = len(CAMERAS)

# <c1,CAM_BACK,1088.3,497.5>  — the id, the camera, then the 2-D image coords.
OBJECT_TAG_RE = re.compile(r"<c\d+\s*,\s*(CAM_[A-Z_]+)\s*,")

# Some DriveLM answers name a camera in prose ("in the CAM_FRONT image") without
# a tag. Catch those too; harmless if absent.
BARE_CAM_RE = re.compile(r"\b(CAM_(?:FRONT|BACK)(?:_LEFT|_RIGHT)?)\b")


def parse_views(text: str, include_bare: bool = True) -> set[str]:
    """Camera names referenced by `text`, as canonical CAMERAS members."""
    if not text:
        return set()
    found = set(OBJECT_TAG_RE.findall(text))
    if include_bare:
        found |= set(BARE_CAM_RE.findall(text))
    return {c for c in found if c in CAM_TO_INDEX}


def evidence_views(
    question: str,
    answer: str,
    source: str = "both",
    include_bare: bool = True,
) -> list[int]:
    """Indices (into CAMERAS) of the views a QA pair depends on.

    `source`:
      "answer"   - only views named in the ground-truth answer
      "question" - only views named in the question
      "both"     - the union (default): a view referenced by the question is just
                   as necessary to look at as one referenced by the answer, since
                   the question grounds its referent there.
    """
    if source == "answer":
        views = parse_views(answer, include_bare)
    elif source == "question":
        views = parse_views(question, include_bare)
    elif source == "both":
        views = parse_views(question, include_bare) | parse_views(answer, include_bare)
    else:
        raise ValueError(f"unknown source: {source!r}")
    return sorted(CAM_TO_INDEX[c] for c in views)


def complement_views(evidence: list[int], num_views: int = NUM_VIEWS) -> list[int]:
    """The views *not* carrying evidence — candidates for the control branch."""
    ev = set(evidence)
    return [i for i in range(num_views) if i not in ev]
