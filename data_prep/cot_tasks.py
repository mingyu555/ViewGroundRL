"""Task adapters for CoT label construction: nuScenes planning and DriveLM QA.

Each adapter supplies three things to `cot_build.py`:

  iter_samples()          raw samples -> {uid, images, gt_answer, context, meta}
  build_prompt(s, hint)   the teacher prompt; `hint` switches to rationalisation
  verify(text, s)         is the generated CoT consistent with the ground truth?

One invariant across both: **`<answer>` is always the ground truth, never the
teacher's**. The teacher is only ever used to produce the reasoning. So a bad
teacher can give us a weak CoT but can never give us a wrong label — which is why
the verification step only has to police the reasoning.

The CoT template asks for ordinary chain-of-thought. It does not ask the model to
declare which camera view its evidence came from; see vgrl/prompt_format.py for
why that matters for the downstream RL stage.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Protocol

from behavior import (
    behavior_from_text,
    behavior_from_waypoints,
    agrees_with,
    parse_waypoints,
)

from vgrl.prompt_format import extract_answer, extract_think
from vgrl.views import CAMERAS, evidence_views

CAPTIONS = [f"{cam}:" for cam in CAMERAS]

COT_STEPS = """Structure your reasoning in four steps:
1. Scene: weather, lighting, road layout, lane markings, and the state of any
   traffic light that governs your direction. Note anything limiting visibility.
2. Key objects: one to three road users or obstacles that matter for safety, where
   they are relative to you, and whether they are stationary, closing, or moving
   away.
3. Reasoning: what the situation and your own state imply about what you should do.
4. Decision: state the manoeuvre plainly."""


class Task(Protocol):
    name: str

    def iter_samples(self) -> Iterator[dict]: ...
    def build_prompt(self, sample: dict, hint: str | None = None) -> str: ...
    def verify(self, text: str, sample: dict) -> tuple[bool, str]: ...


# --------------------------------------------------------------------- nuScenes


class NuScenesPlanningTask:
    """CoT for 3-second trajectory planning.

    The ground-truth trajectory is the answer, and it also *derives* the manoeuvre
    the ego actually performed (data_prep/behavior.py). That gives a free,
    deterministic consistency check on the teacher's conclusion — no judge model
    needed, unlike MindDriver's second API pass.
    """

    name = "nuscenes"

    def __init__(
        self,
        cached_info: str,
        split_json: str,
        split: str,
        nusc_root: str,
        max_samples: int = 0,
        check_images: bool = True,
        ego_status_json: str | None = None,
        require_full_future: bool = True,
        seed: int = 0,
    ):
        import json
        import pickle
        import random
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "create_data"))
        from prompt_message import generate_assistant_message, generate_user_message

        self._gen_assistant = generate_assistant_message
        self._gen_user = generate_user_message
        self.data = pickle.load(open(cached_info, "rb"))
        self.tokens = json.load(open(split_json))[split]
        # Subsample *randomly*, not by prefix: full_split.json is ordered by scene,
        # so taking the first N would train on a handful of scenes.
        if max_samples and max_samples < len(self.tokens):
            self.tokens = sorted(random.Random(seed).sample(self.tokens, max_samples))
        self.nusc_root = nusc_root
        self.check_images = check_images

        # Explicit ego state, as MindDriver's prompt provides. Without it the model
        # has to differentiate the history waypoints itself, and a diagnostic put our
        # first SFT run at UniAD L2 1.50 against 1.51 for a plain constant-
        # acceleration fit of those waypoints — i.e. no benefit from the images.
        self.ego_status = json.load(open(ego_status_json)) if ego_status_json else {}

        # Drop samples whose 3 s future is not fully observed, via nuScenes'
        # `fut_valid_flag` (5,100 of 34,149 samples). Note what this is *not*: the
        # trajectories are never zero-padded, so the all-(0,0) labels in the data are
        # genuinely stationary vehicles (850 samples, 2.5%) and are correct
        # supervision. The first pilot saw 9.2% all-zeros purely because
        # `--max_samples` took a prefix of a scene-ordered split and landed on
        # stationary-heavy scenes — fixed by sampling randomly instead.
        self.require_full_future = require_full_future

    def _reroot(self, paths: list[str]) -> list[str] | None:
        out = []
        for p in paths:
            if "/samples/" not in p:
                return None
            out.append(os.path.join(self.nusc_root, "samples", p.split("/samples/", 1)[1]))
        return out

    def iter_samples(self) -> Iterator[dict]:
        for token in self.tokens:
            if token not in self.data:
                continue

            if self.require_full_future and not self.data[token].get("fut_valid_flag", True):
                continue

            try:
                answer, _stop = self._gen_assistant(self.data, token, traj_only=True)
                context, images_raw = self._gen_user(self.data, token)
            except Exception:
                continue

            images = self._reroot(images_raw)
            if images is None or len(images) != len(CAMERAS):
                continue
            if self.check_images and not all(os.path.exists(p) for p in images):
                continue

            waypoints = parse_waypoints(answer)
            truth = behavior_from_waypoints(waypoints)
            if truth is None:
                continue

            context = context.strip()
            ego = self.ego_status.get(token)
            if ego is not None:
                context += (
                    f"\nCurrent longitudinal speed: {ego['speed']} m/s"
                    f"\nCurrent longitudinal acceleration: {ego['accel']} m/s^2"
                )

            yield {
                "uid": f"nusc::{token}",
                "task": self.name,
                "images": images,
                "captions": CAPTIONS,
                "context": context,
                "gt_answer": answer.strip(),
                "meta": {
                    "token": token,
                    "truth_lateral": truth.lateral,
                    "truth_longitudinal": truth.longitudinal,
                    "speed_start": round(truth.speed_start, 2),
                    "speed_end": round(truth.speed_end, 2),
                    "has_ego_status": ego is not None,
                },
            }

    def build_prompt(self, sample: dict, hint: str | None = None) -> str:
        head = (
            "You are the reasoning module of an autonomous vehicle. Below are the six "
            "surround-view camera images of the current frame, followed by your own "
            "state.\n\n" + sample["context"] + "\n\n"
        )
        if hint is None:
            tail = (
                COT_STEPS + "\n\n"
                "For step 4 choose exactly one lateral action from [go straight, turn "
                "left, turn right, change lane left, change lane right] and exactly one "
                "longitudinal action from [maintain current speed, accelerate, "
                "decelerate, emergency brake, come to a stop, remain stationary]. "
                "Write only the reasoning; do not output any waypoints or coordinates."
            )
        else:
            tail = (
                f"The manoeuvre the vehicle actually performed over the next 3 seconds "
                f"was: {hint}.\n\n" + COT_STEPS + "\n\n"
                "Explain, from what is visible in the images and from your state, why "
                "that is the correct manoeuvre here. Reach exactly that conclusion in "
                "step 4. Do not mention that you were told the answer, and do not "
                "output any waypoints or coordinates."
            )
        return head + tail

    def verify(self, text: str, sample: dict) -> tuple[bool, str]:
        cot = extract_think(text) or text
        if not cot or len(cot.split()) < 20:
            return False, "too_short"
        if parse_waypoints(cot):
            return False, "leaked_coordinates"

        lat, lon = behavior_from_text(cot)
        if lat is None and lon is None:
            return False, "no_decision_stated"

        truth = behavior_from_waypoints(parse_waypoints(sample["gt_answer"]))
        if truth is None:
            return False, "ungradable_gt"
        if not agrees_with(truth, lat, lon):
            return False, f"contradicts_trajectory(said={lat}/{lon},truth={truth.describe()})"
        return True, "ok"

    @staticmethod
    def hint_for(sample: dict) -> str:
        lat = sample["meta"]["truth_lateral"]
        lon = sample["meta"]["truth_longitudinal"]
        # Wording must be phrases behavior.py can match — a slashed form like
        # "turning/moving right" is in neither table, so the teacher would echo it
        # and its own correct answer would then be rejected.
        lat_text = {"straight": "going straight", "left": "turning left",
                    "right": "turning right"}[lat]
        lon_text = {
            "constant": "holding a constant speed",
            "accelerate": "accelerating gently",
            "rapid_accelerate": "accelerating hard",
            "decelerate": "decelerating gently",
            "rapid_decelerate": "braking hard",
            "stop": "slowing to a stop",
            "stationary": "staying stationary",
        }[lon]
        return f"{lat_text} while {lon_text}"


# ---------------------------------------------------------------------- DriveLM


class DriveLMTask:
    """CoT for DriveLM QA.

    Consistency is judged by comparing the teacher's own final answer against the
    reference answer, using the same scoring the RL reward uses so the two stages
    agree on what "right" means.
    """

    name = "drivelm"

    def __init__(
        self,
        drivelm_json: str,
        nusc_root: str,
        categories: str = "perception,prediction,planning,behavior",
        max_samples: int = 0,
        check_images: bool = True,
        f1_threshold: float = 0.45,
        qa_per_frame: str = "",
        seed: int = 0,
    ):
        import json

        self.data = json.load(open(drivelm_json))
        self.nusc_root = nusc_root
        self.categories = set(categories.split(","))
        self.max_samples = max_samples
        self.check_images = check_images
        self.f1_threshold = f1_threshold
        self.seed = seed
        # DriveLM v1.1 train carries ~378k QA over 4072 frames (~93 per frame).
        # Distilling all of them is hours of teacher time for very little extra
        # diversity, so cap per category per frame. Sampling this way keeps every
        # scene represented, which plain truncation would not.
        self.qa_per_frame = self._parse_quota(qa_per_frame)

    @staticmethod
    def _parse_quota(spec: str) -> dict[str, int]:
        """'perception=2,prediction=2,planning=1,behavior=1' -> {cat: n}. Empty = no cap."""
        quota: dict[str, int] = {}
        for part in (spec or "").split(","):
            part = part.strip()
            if not part:
                continue
            cat, _, n = part.partition("=")
            quota[cat.strip()] = int(n)
        return quota

    def _reroot(self, image_paths: dict) -> list[str] | None:
        out = []
        for cam in CAMERAS:
            rel = image_paths.get(cam)
            if not rel or "/samples/" not in rel:
                return None
            out.append(os.path.join(self.nusc_root, "samples", rel.split("/samples/", 1)[1]))
        return out

    def iter_samples(self) -> Iterator[dict]:
        import random

        n = 0
        for scene_token, scene in self.data.items():
            for frame_token, frame in scene.get("key_frames", {}).items():
                images = self._reroot(frame.get("image_paths") or {})
                if images is None:
                    continue
                if self.check_images and not all(os.path.exists(p) for p in images):
                    continue
                for category, items in (frame.get("QA") or {}).items():
                    if category not in self.categories:
                        continue
                    indexed = list(enumerate(items or []))
                    cap = self.qa_per_frame.get(category)
                    if cap is not None and len(indexed) > cap:
                        # seeded on the frame so the subset is stable across runs
                        rng = random.Random(f"{self.seed}:{frame_token}:{category}")
                        indexed = sorted(rng.sample(indexed, cap), key=lambda kv: kv[0])
                    for qa_i, qa in indexed:
                        q, a = qa.get("Q"), qa.get("A")
                        if not q or not a:
                            continue
                        yield {
                            "uid": f"drivelm::{frame_token}::{category}::{qa_i}",
                            "task": self.name,
                            "images": images,
                            "captions": CAPTIONS,
                            "context": q.strip(),
                            "gt_answer": a.strip(),
                            "meta": {
                                "scene_token": scene_token,
                                "frame_token": frame_token,
                                "category": category,
                                "evidence_views": evidence_views(q, a, source="both"),
                            },
                        }
                        n += 1
                        if self.max_samples and n >= self.max_samples:
                            return

    def build_prompt(self, sample: dict, hint: str | None = None) -> str:
        head = (
            "You are the perception and reasoning module of an autonomous vehicle. "
            "Below are the six surround-view camera images of the current frame.\n\n"
            "Objects may be referred to as <cID,CAMERA,x,y>, where CAMERA names the "
            "view the object appears in and (x,y) is its pixel location in that view.\n\n"
            f"Question: {sample['context']}\n\n"
        )
        if hint is None:
            tail = (
                "Think it through step by step: describe what is relevant in the scene, "
                "identify the objects that bear on the question and their state, then "
                "reason to a conclusion.\n\n"
                "Format your response as:\nREASONING: <your step-by-step reasoning>\n"
                "ANSWER: <your final answer>"
            )
        else:
            tail = (
                f"The correct answer is: {hint}\n\n"
                "Explain, step by step and only from what is visible in the images, the "
                "reasoning that leads to that answer: describe what is relevant in the "
                "scene, identify the objects that bear on the question and their state, "
                "then reason to the conclusion. Do not mention that you were given the "
                "answer.\n\n"
                "Format your response as:\nREASONING: <your step-by-step reasoning>\n"
                f"ANSWER: {hint}"
            )
        return head + tail

    @staticmethod
    def split_reasoning_answer(text: str) -> tuple[str, str]:
        """Pull REASONING / ANSWER out of the teacher response."""
        t = text or ""
        m = re.search(r"REASONING\s*:\s*(.*?)(?:\n\s*ANSWER\s*:|\Z)", t, re.S | re.I)
        reasoning = (m.group(1).strip() if m else "")
        m2 = re.search(r"ANSWER\s*:\s*(.*)", t, re.S | re.I)
        answer = (m2.group(1).strip() if m2 else "")
        if not reasoning and not answer:
            # no headings at all: treat the whole thing as reasoning
            reasoning = t.strip()
        return reasoning, answer

    def verify(self, text: str, sample: dict) -> tuple[bool, str]:
        from vgrl.rewards import (
            _f1,
            behavior_reward,
            multiple_choice_reward,
            object_reference_reward,
        )

        reasoning, answer = self.split_reasoning_answer(text)
        if not reasoning or len(reasoning.split()) < 15:
            return False, "too_short"
        if not answer:
            return False, "no_answer_stated"

        gt = sample["gt_answer"]
        category = sample["meta"]["category"]

        if category == "behavior":
            ok = behavior_reward([answer], solution=[gt], category=["behavior"])[0] == 1.0
            return (ok, "ok" if ok else "behavior_mismatch")

        if "select the best answer" in sample["context"].lower() or re.search(
            r"\b[A-D]\s*\.", sample["context"]
        ):
            ok = multiple_choice_reward([answer], solution=[gt])[0] == 1.0
            return (ok, "ok" if ok else "wrong_option")

        f1 = _f1(answer, gt)
        obj = object_reference_reward([answer], solution=[gt])[0]
        # object-reference credit can carry a sample whose wording differs a lot but
        # which cites exactly the right objects
        ok = f1 >= self.f1_threshold or obj >= 0.5
        return (ok, "ok" if ok else f"low_agreement(f1={f1:.2f},obj={obj:.2f})")

    @staticmethod
    def hint_for(sample: dict) -> str:
        return sample["gt_answer"]


def build_task(kind: str, **kwargs) -> Task:
    if kind == "nuscenes":
        return NuScenesPlanningTask(**kwargs)
    if kind == "drivelm":
        return DriveLMTask(**kwargs)
    raise ValueError(f"unknown task: {kind!r}")
