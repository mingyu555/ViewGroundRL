"""Stage 1: plain SFT on the teacher-distilled CoT set.

    accelerate launch --num_processes 4 train_sft.py --config configs/sft_trl.yaml

Uses TRL's SFTTrainer rather than LLaMA-Factory. Two reasons: the LLaMA-Factory
copy inherited from MindDriver is gutted (58 of 64 source files are 0 bytes), and
the real package pins `transformers<=4.51`, which cannot coexist with the
transformers 5.x / TRL 1.9.2 stack the RL stage needs. Sharing one stack across
both stages means no environment switch between them, and the SFT checkpoint drops
straight into `configs/rl_drivelm.yaml`.

No view-grounding term here by design — see docs/METHOD.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vgrl.prompt_format import ANSWER_OPEN, THINK_OPEN  # noqa: E402


def to_conversation(row: dict) -> dict:
    """sharegpt row -> TRL *prompt-completion* format with image placeholders.

    The stored `value` already contains one `<image>` per view (VIEW_HEADER); TRL
    wants those as separate content parts, so the text is split on the placeholder
    and the pieces interleaved with `{"type": "image"}` entries.

    Prompt and completion are kept in separate fields rather than a single
    `messages` list, because TRL only honours `completion_only_loss` for
    prompt-completion data — with `messages` it raises, and the alternative
    (`assistant_only_loss`) needs `{% generation %}` markers that the Qwen2.5-VL
    chat template does not have.
    """
    system = row.get("system")
    convs = row["conversations"]
    user_text = next(c["value"] for c in convs if c["from"] == "human")
    assistant_text = next(c["value"] for c in convs if c["from"] == "gpt")

    parts: list[dict] = []
    chunks = user_text.split("<image>")
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if i < len(chunks) - 1:
            parts.append({"type": "image"})

    n_img = len(chunks) - 1
    if n_img != len(row["images"]):
        raise ValueError(
            f"{row.get('id')}: {n_img} <image> placeholders but {len(row['images'])} images"
        )

    prompt = []
    if system:
        prompt.append({"role": "system", "content": [{"type": "text", "text": system}]})
    prompt.append({"role": "user", "content": parts})
    completion = [{"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
    return {"prompt": prompt, "completion": completion, "images": row["images"]}


def load_dataset(path: str, max_rows: int = 0):
    from datasets import Dataset
    from PIL import Image

    rows = json.load(open(path))
    if max_rows:
        rows = rows[:max_rows]

    bad = 0
    converted = []
    for r in rows:
        try:
            converted.append(to_conversation(r))
        except ValueError:
            bad += 1
    if bad:
        print(f"  {path}: dropped {bad} rows with mismatched image placeholders")

    ds = Dataset.from_list(converted)

    def to_pil(batch):
        batch["images"] = [
            [Image.open(p).convert("RGB") for p in imgs] for imgs in batch["images"]
        ]
        return batch

    ds.set_transform(to_pil)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
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
    max_train_rows = cfg.pop("max_train_rows", 0)
    max_eval_rows = cfg.pop("max_eval_rows", 0)
    attn_impl = cfg.pop("attn_implementation", "sdpa")
    freeze_vision = cfg.pop("freeze_vision_tower", True)
    # 7B 는 전체 FT 가 ~106GB (가중치 15 + grad 30 + AdamW 61) 라 공유 GPU 에서
    # 돌지 않는다. LoRA 로 간다. 3B 결과는 전체 FT 였으므로 3B<->7B 비교에는
    # 학습 방식 차이가 섞인다는 점을 명시해야 한다.
    lora_cfg = cfg.pop("lora", None)
    # Not SFTConfig fields — these belong to the image processor. With six views per
    # sample the visual-token budget is what decides whether a sequence fits in
    # max_length, so it has to be pinned explicitly rather than left at the
    # processor default (12.8M px, i.e. ~4k tokens *per view*).
    max_pixels = cfg.pop("max_pixels", None)
    min_pixels = cfg.pop("min_pixels", None)

    from transformers import AutoProcessor
    from trl import SFTConfig, SFTTrainer

    # The pixel caps have to go through from_pretrained. Setting them afterwards
    # silently does nothing: `image_processor.max_pixels` is not what smart_resize
    # reads, and `image_processor.size` is a SizeDict rather than a dict, so an
    # isinstance(dict) guard skips it. Left unset, each 1600x900 view costs 1824
    # visual tokens and six views alone blow past max_length.
    proc_kwargs = {}
    if max_pixels:
        proc_kwargs["max_pixels"] = max_pixels
    if min_pixels:
        proc_kwargs["min_pixels"] = min_pixels
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **proc_kwargs)
    if proc_kwargs:
        print(f"image processor size: {getattr(processor.image_processor, 'size', None)}")

    train_ds = load_dataset(train_file, max_train_rows)
    eval_ds = load_dataset(eval_file, max_eval_rows) if eval_file else None
    print(f"train rows: {len(train_ds)}" + (f"  eval rows: {len(eval_ds)}" if eval_ds else ""))

    # In TRL 1.9 the model kwargs live on the config, not the trainer signature.
    cfg.setdefault("model_init_kwargs", {})
    cfg["model_init_kwargs"].update({"attn_implementation": attn_impl, "dtype": "bfloat16"})

    sft_args = SFTConfig(**cfg)
    # Train on the response only: the prompt carries ~6 views' worth of image
    # placeholders and a long fixed instruction block, and letting the loss cover
    # them would drown the CoT signal we actually want.
    if getattr(sft_args, "completion_only_loss", None) is None:
        sft_args.completion_only_loss = True

    peft_config = None
    if lora_cfg:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        print(f"LoRA r={peft_config.r} alpha={peft_config.lora_alpha} "
              f"targets={peft_config.target_modules}")

    trainer = SFTTrainer(
        model=model_id,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    if freeze_vision:
        # ZeRO-3 는 파라미터를 샤딩해서 named_parameters() 가 빈 텐서를 돌려준다.
        # 그대로 두면 requires_grad 를 못 바꿔 vision tower 가 학습돼 LoRA 판과
        # 조건이 달라진다 (실측: "froze vision tower: 0.0M params").
        # ds_numel 로 원래 크기를 읽고, requires_grad 는 샤드에도 정상 반영된다.
        frozen = 0
        for name, param in trainer.model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad = False
                frozen += getattr(param, "ds_numel", param.numel())
        print(f"froze vision tower: {frozen/1e6:.1f}M params")

    # ZeRO-3 는 파라미터를 랭크별로 쪼개서 named_parameters() 가 빈 껍데기를 돌려준다.
    # numel() 이 0 이 되므로 원래 크기가 담긴 ds_numel 을 먼저 본다 (197행과 같은 처방).
    trainable = sum(getattr(p, "ds_numel", p.numel())
                    for p in trainer.model.parameters() if p.requires_grad)
    total = sum(getattr(p, "ds_numel", p.numel()) for p in trainer.model.parameters())
    print(f"trainable params: {trainable/1e9:.2f}B / {total/1e9:.2f}B "
          f"({100*trainable/max(total,1):.1f}%)")

    # resume_from_checkpoint 는 SFTConfig 필드로 넘겨도 무시된다 — HF Trainer 는
    # 이 값을 train() 의 인자로 받는다. 이걸 몰라 두 번(중단된 866/1300 스텝) 재개에
    # 실패하고 처음부터 다시 돌렸다.
    resume = getattr(sft_args, "resume_from_checkpoint", None)
    if resume:
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(sft_args.output_dir)
    processor.save_pretrained(sft_args.output_dir)
    print(f"saved to {sft_args.output_dir}")

    # a quick sanity read on the format the student now emits
    print(f"(SFT targets wrap reasoning in {THINK_OPEN} and the answer in {ANSWER_OPEN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
