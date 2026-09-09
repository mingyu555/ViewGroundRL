"""DriveLM 용 VGGDrive stage 1 — VLM 을 얼린 채 CVGE 만 학습한다.

train_sft_cvge.py 는 DriveLMM-o1 용(2x3 격자 한 장)이라 입력 구성이 다르다.
DriveLM 은 6뷰를 따로 넣으므로:
  - Qwen  : 6뷰 개별 이미지 (평가와 동일)
  - VGGT  : 같은 6뷰 원본, 518 폭으로 리사이즈
교차 어텐션이라 토큰 수가 맞을 필요는 없다.

논문 4.1: VGGT 는 두 단계 내내 동결, stage 1 은 base VLM 도 동결하고 CVGE 만
2 epoch / lr 1e-4 로 학습.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_sft_cvge import vggt_preprocess  # noqa: E402
from vgrl.cvge import CVGE, load_vggt, vggt_features  # noqa: E402


class DlmCvgeDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_rows: int = 0):
        rows = json.load(open(path))
        self.rows = rows[:max_rows] if max_rows else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "images": r["images"],
            "system": r["system"],
            "question": r["question"],
            "target": r["conversations"][-1]["value"],
        }


class Collator:
    def __init__(self, processor, max_length: int, max_pixels: int):
        self.processor = processor
        self.max_length = max_length
        self.max_pixels = max_pixels

    def __call__(self, batch: list[dict]) -> dict:
        texts, images, vggt_imgs, prompts = [], [], [], []
        for ex in batch:
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": ex["system"]}]},
                {"role": "user", "content": [{"type": "image"} for _ in ex["images"]]
                 + [{"type": "text", "text": ex["question"]}]},
            ]
            p = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
            texts.append(p + ex["target"] + "<|im_end|>\n")
            images.append([Image.open(x).convert("RGB") for x in ex["images"]])
            vggt_imgs.append(vggt_preprocess(ex["images"]))

        enc = self.processor(text=texts, images=images, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, p in enumerate(prompts):
            n = self.processor(text=[p], images=[images[i]],
                               return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n] = -100
        enc["labels"] = labels
        enc["vggt_images"] = torch.stack(vggt_imgs)
        return enc


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
    vggt_weights = cfg.pop("vggt_weights")
    train_file = cfg.pop("train_file")
    eval_file = cfg.pop("eval_file", None)
    max_train_rows = cfg.pop("max_train_rows", 0)
    max_eval_rows = cfg.pop("max_eval_rows", 0)
    attn_impl = cfg.pop("attn_implementation", "sdpa")
    max_pixels = cfg.pop("max_pixels", 401408)
    min_pixels = cfg.pop("min_pixels", None)
    max_length = cfg.pop("max_length", 4096)
    cvge_cfg = cfg.pop("cvge", {}) or {}
    debug_nan = cfg.pop("debug_nan", False)

    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    proc_kwargs = {"max_pixels": max_pixels}
    if min_pixels:
        proc_kwargs["min_pixels"] = min_pixels
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **proc_kwargs)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.requires_grad_(False)          # stage 1: VLM 전체 동결
    model.config.use_cache = False

    tc = getattr(model.config, "text_config", model.config)
    hidden, n_layers = tc.hidden_size, tc.num_hidden_layers
    cvge = CVGE(hidden_size=hidden, num_layers=n_layers,
                in_dim=cvge_cfg.get("in_dim", 2048),
                scale=cvge_cfg.get("scale", 4),
                num_heads=cvge_cfg.get("num_heads", 8),
                dropout=cvge_cfg.get("dropout", 0.1))
    cvge.attach(model.model.language_model.layers)
    model.cvge = cvge

    n_train = sum(p.numel() for p in cvge.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"CVGE 학습 파라미터 {n_train/1e6:.1f}M / 전체 {n_total/1e9:.2f}B "
          f"({100*n_train/n_total:.2f}%)  [hidden {hidden}, layers {n_layers}]")

    image_token_id = model.config.image_token_id
    vggt = None

    class CvgeTrainer(Trainer):
        _warned = False

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            nonlocal vggt
            imgs = inputs.pop("vggt_images")
            dev = next(model.parameters()).device
            if vggt is None:
                vggt = load_vggt(vggt_weights, device=dev, dtype=torch.bfloat16)
            f3d = vggt_features(vggt, imgs.to(device=dev, dtype=torch.bfloat16))
            img_mask = inputs["input_ids"] == image_token_id
            core = model.module if hasattr(model, "module") else model
            core.cvge.set_batch(f3d, img_mask)
            try:
                out = model(**inputs)
            finally:
                core.cvge.clear_batch()
            return (out.loss, out) if return_outputs else out.loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            # 비유한 그래디언트를 clip 전에 버린다. bf16 은 GradScaler 가 없어서
            # inf 가 clip_grad_norm_ 에 들어가면 1.0/inf = 0, inf*0 = nan 으로
            # 전 파라미터가 죽는다 (DriveLMM-o1 CVGE 학습에서 실제로 발생).
            n_bad = 0
            for p_ in self.model.parameters():
                if p_.grad is not None and not torch.isfinite(p_.grad).all():
                    p_.grad.zero_(); n_bad += 1
            if n_bad and not self._warned:
                self._warned = True
                print(f"[step {self.state.global_step}] 비유한 그래디언트 {n_bad}개 폐기",
                      flush=True)
            return loss

    train_ds = DlmCvgeDataset(train_file, max_train_rows)
    eval_ds = DlmCvgeDataset(eval_file, max_eval_rows) if eval_file else None
    print(f"train rows: {len(train_ds)}" + (f"  eval rows: {len(eval_ds)}" if eval_ds else ""))

    targs = TrainingArguments(**cfg)
    trainer = CvgeTrainer(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=eval_ds,
                          data_collator=Collator(processor, max_length, max_pixels))
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))

    if trainer.is_world_process_zero():
        os.makedirs(targs.output_dir, exist_ok=True)
        core = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        torch.save(core.cvge.state_dict(), os.path.join(targs.output_dir, "cvge.pt"))
        json.dump({"model": model_id, "vggt_weights": vggt_weights,
                   "hidden_size": hidden, "num_layers": n_layers, **cvge_cfg},
                  open(os.path.join(targs.output_dir, "cvge_config.json"), "w"), indent=2)
        processor.save_pretrained(targs.output_dir)
        print(f"saved CVGE -> {targs.output_dir}/cvge.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
