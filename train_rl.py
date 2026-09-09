"""Entry point for view-grounded GRPO on DriveLM.

    python train_rl.py --config configs/rl_drivelm.yaml

Everything GRPO-specific is delegated to TRL's GRPOConfig; keys under
`view_grounding:` in the yaml populate vgrl.ViewGroundingConfig.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vgrl.config import ViewGroundingConfig  # noqa: E402
from vgrl.rewards import build_rewards  # noqa: E402


def load_dataset(path: str, keep_unusable: bool = True):
    from datasets import Dataset

    rows = json.load(open(path))
    if not keep_unusable:
        rows = [r for r in rows if r.get("vg_usable", True)]
    # `images` must be a list of paths/PIL images; datasets keeps it as-is when we
    # build from a python list, and TRL's processor handles path loading.
    from PIL import Image

    def to_pil(batch):
        # set_transform hands us a *batch*, so images is a list of per-row lists
        # (six views each) — not a flat list of paths.
        batch["images"] = [
            [Image.open(p).convert("RGB") for p in row] for row in batch["images"]
        ]
        return batch

    ds = Dataset.from_list(rows)
    ds.set_transform(to_pil)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="key=value pairs applied on top of the yaml, e.g. "
                         "view_grounding.coef=0.05")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)

    model_id = cfg.pop("model")
    train_file = cfg.pop("train_file")
    eval_file = cfg.pop("eval_file", None)
    reward_names = cfg.pop("rewards", ["text_overlap", "object_reference", "format"])
    reward_weights = cfg.pop("reward_weights", None)
    vg_cfg_dict = cfg.pop("view_grounding", {}) or {}
    keep_unusable = cfg.pop("keep_unusable_rows", True)
    attn_impl = cfg.pop("attn_implementation", "flash_attention_2")
    # Not GRPOConfig fields; they must reach AutoProcessor.from_pretrained, because
    # setting them on the image processor afterwards is a silent no-op (its `size`
    # is a SizeDict, and `max_pixels` is not the attribute smart_resize reads).
    # Unset, each 1600x900 view costs 1824 visual tokens; six views blow the budget.
    max_pixels = cfg.pop("max_pixels", None)
    min_pixels = cfg.pop("min_pixels", None)

    from transformers import AutoProcessor
    from trl import GRPOConfig

    from vgrl.grpo_view_trainer import ViewGroundedGRPOTrainer

    vg_config = ViewGroundingConfig(**vg_cfg_dict)
    vg_config.validate()

    # In TRL 1.9 the model kwargs live on the config, not the trainer signature —
    # same as SFTTrainer. Passing them to the trainer raises TypeError.
    cfg.setdefault("model_init_kwargs", {})
    cfg["model_init_kwargs"].update({"attn_implementation": attn_impl, "dtype": "bfloat16"})

    # 7B 는 전체 FT 가 ~106GB 라 공유 GPU 에서 불가. SFT 와 같은 LoRA 설정을 쓴다.

    cfg_resume = cfg.pop("resume_from_checkpoint", None)
    lora_cfg = cfg.pop("lora", None)

    peft_config = None

    if lora_cfg:

        from peft import LoraConfig

        peft_config = LoraConfig(

            r=lora_cfg.get("r", 64), lora_alpha=lora_cfg.get("alpha", 16),

            lora_dropout=lora_cfg.get("dropout", 0.05), bias="none",

            task_type="CAUSAL_LM",

            target_modules=lora_cfg.get("target_modules",

                                        ["q_proj", "k_proj", "v_proj", "o_proj"]))

        print(f"LoRA r={peft_config.r} targets={peft_config.target_modules}")

    grpo_args = GRPOConfig(**cfg)
    if reward_weights is not None:
        grpo_args.reward_weights = reward_weights

    proc_kwargs = {}
    if max_pixels:
        proc_kwargs["max_pixels"] = max_pixels
    if min_pixels:
        proc_kwargs["min_pixels"] = min_pixels
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **proc_kwargs)
    if proc_kwargs:
        print(f"image processor size: {getattr(processor.image_processor, 'size', None)}")
    train_ds = load_dataset(train_file, keep_unusable=keep_unusable)
    eval_ds = load_dataset(eval_file, keep_unusable=keep_unusable) if eval_file else None

    print(f"train rows: {len(train_ds)}" + (f"  eval rows: {len(eval_ds)}" if eval_ds else ""))
    print(f"view grounding: {vg_config}")

    trainer = ViewGroundedGRPOTrainer(
        model=model_id,
        reward_funcs=build_rewards(reward_names),
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        vg_config=vg_config,
        peft_config=peft_config,
    )
    # 중간에 죽어도 100스텝만 잃도록 재개를 지원한다 (실측: 240/500 에서 외부 종료로
    # 5.4시간을 통째로 잃었다. save_strategy="no" 였어서 아무것도 남지 않았다).
    resume = cfg_resume or None
    if resume:
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(grpo_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
