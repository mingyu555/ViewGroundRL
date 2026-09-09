"""Build CoT SFT labels with a teacher VLM.

    # nuScenes planning
    python data_prep/cot_build.py --task nuscenes \
        --cached_info /mnt/ssd1/vgrl/raw/cached_nuscenes_info.pkl \
        --split_json create_data/full_split.json --split train \
        --nusc_root /nuscenes \
        --work_dir /mnt/ssd1/vgrl/cot/nuscenes \
        --out_json /mnt/ssd1/vgrl/data/nuscenes_cot_sft.json

    # DriveLM QA
    python data_prep/cot_build.py --task drivelm \
        --drivelm_json /mnt/ssd1/vgrl/raw/v1_1_train_nus.json \
        --nusc_root /nuscenes \
        --work_dir /mnt/ssd1/vgrl/cot/drivelm \
        --out_json /mnt/ssd1/vgrl/data/drivelm_cot_sft.json

Two passes, following the STaR/rationalisation recipe:

  pass 1  generate reasoning *without* showing the teacher the answer, and keep
          only the samples whose reasoning is consistent with the ground truth;
  pass 2  for the rest, regenerate *with* the answer supplied and ask the teacher
          to justify it.

Pass 1 yields genuine reasoning; pass 2 recovers the hard samples instead of
throwing them away, and is flagged in the output (`cot_pass`) so its share can be
measured or capped. `<answer>` is the ground truth in both passes, so no pass can
introduce a wrong label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cot_tasks import DriveLMTask, NuScenesPlanningTask, build_task  # noqa: E402
from cot_teacher import TeacherConfig, TeacherRequest, check_endpoint, generate  # noqa: E402

from vgrl.prompt_format import (  # noqa: E402
    SYSTEM_DRIVELM,
    SYSTEM_NUSCENES,
    NUSCENES_INSTRUCTION,
    VIEW_HEADER,
    extract_think,
    mentions_camera,
    strip_camera_names,
    wrap,
)


def teacher_cot(task, text: str) -> str:
    """The reasoning the teacher produced, whatever wrapper it used."""
    if isinstance(task, DriveLMTask):
        reasoning, _ = task.split_reasoning_answer(text)
        return reasoning
    return extract_think(text) or (text or "").strip()


def sft_row(task, sample: dict, cot: str, cot_pass: int) -> dict:
    if isinstance(task, NuScenesPlanningTask):
        system = SYSTEM_NUSCENES
        user = VIEW_HEADER + sample["context"] + "\n" + NUSCENES_INSTRUCTION
    else:
        system = SYSTEM_DRIVELM
        user = VIEW_HEADER + sample["context"]

    row = {
        "id": sample["uid"],
        "images": sample["images"],
        "system": system,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": wrap(cot, sample["gt_answer"])},
        ],
        "cot_pass": cot_pass,
        "task": sample["task"],
    }
    # carried through for the RL stage; harmless for SFT
    if "evidence_views" in sample["meta"]:
        row["evidence_views"] = sample["meta"]["evidence_views"]
    if "category" in sample["meta"]:
        row["category"] = sample["meta"]["category"]
    return row


def make_requests(task, samples, hints: dict[str, str] | None = None):
    for s in samples:
        hint = hints.get(s["uid"]) if hints else None
        yield TeacherRequest(
            uid=s["uid"],
            text=task.build_prompt(s, hint=hint),
            images=s["images"],
            captions=s["captions"],
            meta={"task": s["task"]},
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["nuscenes", "drivelm"])
    ap.add_argument("--work_dir", required=True, help="where the teacher jsonl caches live")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--nusc_root", required=True)

    # nuScenes
    ap.add_argument("--cached_info")
    ap.add_argument("--split_json")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    # DriveLM
    ap.add_argument("--drivelm_json")
    ap.add_argument("--categories", default="perception,prediction,planning,behavior")
    ap.add_argument("--f1_threshold", type=float, default=0.45)
    ap.add_argument("--qa_per_frame", default="",
                    help="cap DriveLM QA per category per frame, e.g. "
                         "'perception=2,prediction=2,planning=1,behavior=1'. Sampling is "
                         "seeded on the frame token so the subset is stable across runs. "
                         "Empty means take everything (~378k QA — hours of teacher time).")

    ap.add_argument("--max_samples", type=int, default=0,
                    help="random subsample (seeded), not a prefix — full_split.json is "
                         "ordered by scene so a prefix would cover few scenes")
    ap.add_argument("--ego_status_json", default=None,
                    help="speed/acceleration per sample token (data_prep/ego_status.py); "
                         "matches the explicit ego state MindDriver's prompt provides")
    ap.add_argument("--allow_partial_future", action="store_true",
                    help="keep samples whose 3s future is only partly observed. Off by "
                         "default: their padded (0,0) steps taught the first run to "
                         "answer 'stay put'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_check_images", action="store_true")

    # teacher
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL",
                                                      "Qwen/Qwen2.5-VL-72B-Instruct"))
    ap.add_argument("--base_url", default=os.environ.get("TEACHER_BASE_URL",
                                                         "http://127.0.0.1:8000/v1"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--image_long_edge", type=int, default=896)

    # label policy
    ap.add_argument("--no_hint_pass", action="store_true",
                    help="skip pass 2; drop inconsistent samples instead")
    ap.add_argument("--max_hint_fraction", type=float, default=1.0,
                    help="cap the share of rows that come from pass 2")
    ap.add_argument("--forbid_camera_mentions", action="store_true",
                    help="rewrite camera names out of the CoT. Off by default: the "
                         "reasoning is allowed to mention a view in passing, it is "
                         "only never *asked* to attribute the answer to one. Turning "
                         "this on for DriveLM is usually counterproductive, since its "
                         "reference answers cite <cID,CAMERA,...> themselves.")
    ap.add_argument("--dry_run", type=int, default=0,
                    help="build N requests, print the first prompt, and stop")
    args = ap.parse_args()

    check_images = not args.no_check_images
    if args.task == "nuscenes":
        missing = [f for f in ("cached_info", "split_json") if not getattr(args, f)]
        if missing:
            raise SystemExit(f"--task nuscenes needs {missing}")
        task = build_task(
            "nuscenes",
            cached_info=args.cached_info,
            split_json=args.split_json,
            split=args.split,
            nusc_root=args.nusc_root,
            max_samples=args.max_samples,
            check_images=check_images,
            ego_status_json=args.ego_status_json,
            require_full_future=not args.allow_partial_future,
            seed=args.seed,
        )
    else:
        if not args.drivelm_json:
            raise SystemExit("--task drivelm needs --drivelm_json")
        task = build_task(
            "drivelm",
            drivelm_json=args.drivelm_json,
            nusc_root=args.nusc_root,
            categories=args.categories,
            max_samples=args.max_samples,
            check_images=check_images,
            f1_threshold=args.f1_threshold,
            qa_per_frame=args.qa_per_frame,
        )

    samples = list(task.iter_samples())
    print(f"{args.task}: {len(samples)} gradable samples")
    if not samples:
        raise SystemExit("nothing to do — check the input paths and --nusc_root")

    if args.dry_run:
        for s in samples[: args.dry_run]:
            print("=" * 70)
            print(f"uid: {s['uid']}")
            print(f"images: {len(s['images'])}  meta: {s['meta']}")
            print("--- prompt (pass 1) ---")
            print(task.build_prompt(s))
            print("--- gt answer ---")
            print(s["gt_answer"][:400])
            print("--- prompt (pass 2, hinted) ---")
            print(task.build_prompt(s, hint=task.hint_for(s))[:1200])
        return 0

    cfg = TeacherConfig(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image_long_edge=args.image_long_edge,
    )
    print(check_endpoint(cfg))
    os.makedirs(args.work_dir, exist_ok=True)

    by_uid = {s["uid"]: s for s in samples}
    stats = Counter()

    # ---------------------------------------------------------------- pass 1
    print("\n### pass 1: unhinted reasoning")
    p1_path = os.path.join(args.work_dir, "pass1.jsonl")
    results1 = generate(make_requests(task, samples), cfg, p1_path)

    rows: list[dict] = []
    needs_hint: list[dict] = []
    for uid, sample in by_uid.items():
        rec = results1.get(uid)
        if not rec or not rec.get("ok"):
            stats["pass1/api_failed"] += 1
            needs_hint.append(sample)
            continue
        ok, reason = task.verify(rec["text"], sample)
        if ok:
            cot = teacher_cot(task, rec["text"])
            rows.append(sft_row(task, sample, cot, cot_pass=1))
            stats["pass1/accepted"] += 1
        else:
            stats[f"pass1/rejected:{reason.split('(')[0]}"] += 1
            needs_hint.append(sample)

    print(f"pass 1 accepted {len(rows)}/{len(samples)}; {len(needs_hint)} need pass 2")

    # ---------------------------------------------------------------- pass 2
    if needs_hint and not args.no_hint_pass:
        print("\n### pass 2: rationalise the known answer")
        p2_path = os.path.join(args.work_dir, "pass2.jsonl")
        hints = {s["uid"]: task.hint_for(s) for s in needs_hint}
        results2 = generate(make_requests(task, needs_hint, hints), cfg, p2_path)

        hint_rows: list[dict] = []
        for sample in needs_hint:
            rec = results2.get(sample["uid"])
            if not rec or not rec.get("ok"):
                stats["pass2/api_failed"] += 1
                continue
            ok, reason = task.verify(rec["text"], sample)
            if not ok:
                stats[f"pass2/rejected:{reason.split('(')[0]}"] += 1
                continue
            cot = teacher_cot(task, rec["text"])
            hint_rows.append(sft_row(task, sample, cot, cot_pass=2))
            stats["pass2/accepted"] += 1

        cap = int(len(rows) * args.max_hint_fraction / max(1 - args.max_hint_fraction, 1e-9)) \
            if args.max_hint_fraction < 1.0 else len(hint_rows)
        if len(hint_rows) > cap:
            stats["pass2/dropped_by_cap"] += len(hint_rows) - cap
            hint_rows = hint_rows[:cap]
        rows.extend(hint_rows)
    elif needs_hint:
        stats["dropped_no_hint_pass"] += len(needs_hint)

    # ---------------------------------------------------------------- finish
    if args.forbid_camera_mentions:
        n = 0
        for r in rows:
            body = r["conversations"][1]["value"]
            if mentions_camera(extract_think(body) or ""):
                think = strip_camera_names(extract_think(body) or "")
                answer = body.split("<answer>", 1)[1].rsplit("</answer>", 1)[0]
                r["conversations"][1]["value"] = wrap(think, answer)
                n += 1
        stats["camera_names_stripped"] = n

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"\nwrote {args.out_json}: {len(rows)} rows")
    print("--- stats ---")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if rows:
        n2 = sum(1 for r in rows if r["cot_pass"] == 2)
        print(f"  pass-2 share of output: {n2}/{len(rows)} = {n2 / len(rows):.1%}")
        lens = [len((extract_think(r["conversations"][1]["value"]) or "").split()) for r in rows]
        lens.sort()
        print(f"  CoT length words: p10={lens[len(lens)//10]} "
              f"median={lens[len(lens)//2]} p90={lens[9*len(lens)//10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
