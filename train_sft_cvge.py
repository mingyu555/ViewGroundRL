"""VGGDrive stage 1 — VLM 을 얼린 채 CVGE 만 학습한다.

논문 4.1:
  "In the first stage, we freeze the base VLM parameters and train only the
   parameters introduced by CVGE for 2 epochs, using a learning rate of 1e-4
   and a batch size of 2."

VGGT 는 두 단계 내내 얼려 있다. 여기서는 stage 1 만 돈다.

DriveLMM-o1 에 맞춘 부분:
  - Qwen 은 지금까지와 같은 2x3 격자 한 장을 본다. 기존 SFT 기준선(MCQ 64.24%)과
    입력을 맞춰야 "VGGT 를 붙여서 얼마나 달라지나"가 분리된다.
  - VGGT 는 격자를 만들기 전의 6 뷰 원본을 본다. 교차 어텐션이라 토큰 수가 맞을
    필요가 없어서, 격자 토큰 2752 개가 VGGT 토큰 4692 개를 참조하는 형태가 된다.
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

from vgrl.cvge import CVGE, load_vggt, vggt_features  # noqa: E402

VGGT_WIDTH = 518          # VGGT load_fn 의 crop 모드 기본값
PATCH = 14


def vggt_preprocess(paths: list[str], width: int = VGGT_WIDTH) -> torch.Tensor:
    """VGGT load_and_preprocess_images(mode='crop') 와 같은 전처리.

    ToTensor 만 쓴다 — VGGT 는 ImageNet 정규화를 하지 않는다. 1600x900 은
    518x294 가 된다(높이를 14 의 배수로 반올림).
    """
    import torchvision.transforms.functional as TF

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_h = round(h * (width / w) / PATCH) * PATCH
        img = img.resize((width, new_h), Image.Resampling.BICUBIC)
        t = TF.to_tensor(img)                       # [3, H, W], 0..1
        if new_h > width:                           # crop 모드: 세로 중앙 크롭
            s = (new_h - width) // 2
            t = t[:, s:s + width, :]
        out.append(t)
    return torch.stack(out)                         # [S, 3, H, W]


class DlmmCvgeDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_rows: int = 0):
        rows = json.load(open(path))
        self.rows = rows[:max_rows] if max_rows else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        target = r["conversations"][-1]["value"]
        return {
            "grid": r["images"][0],
            "views": r["views"],
            "system": r["system"],
            "user": r["prompt_text"],
            "target": target,
        }


class Collator:
    """텍스트/격자는 Qwen 프로세서로, 6 뷰는 VGGT 전처리로 따로 만든다."""

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts, images, vggt_imgs = [], [], []
        prompt_lens = []
        for ex in batch:
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": ex["system"]}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["user"]},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            texts.append(prompt + ex["target"] + "<|im_end|>\n")
            prompt_lens.append(prompt)
            images.append(Image.open(ex["grid"]).convert("RGB"))
            vggt_imgs.append(vggt_preprocess(ex["views"]))

        enc = self.processor(
            text=texts, images=images, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        # 프롬프트 구간은 손실에서 뺀다. 격자 이미지 토큰 2752 개와 고정 지시문이
        # 손실을 덮으면 CoT 신호가 묻힌다 (기존 SFT 와 같은 방침).
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, p in enumerate(prompt_lens):
            n_prompt = self.processor(
                text=[p], images=[images[i]], return_tensors="pt",
            )["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        enc["vggt_images"] = torch.stack(vggt_imgs)      # [B, S, 3, H, W]
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
    max_pixels = cfg.pop("max_pixels", None)
    min_pixels = cfg.pop("min_pixels", None)
    max_length = cfg.pop("max_length", 4096)
    cvge_cfg = cfg.pop("cvge", {}) or {}

    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Trainer, TrainingArguments)

    proc_kwargs = {}
    if max_pixels:
        proc_kwargs["max_pixels"] = max_pixels
    if min_pixels:
        proc_kwargs["min_pixels"] = min_pixels
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **proc_kwargs)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn_impl,
    )
    # stage 1: VLM 전체 동결
    model.requires_grad_(False)
    model.config.use_cache = False

    hidden = model.config.text_config.hidden_size
    n_layers = model.config.text_config.num_hidden_layers
    cvge = CVGE(
        hidden_size=hidden, num_layers=n_layers,
        in_dim=cvge_cfg.get("in_dim", 2048),
        scale=cvge_cfg.get("scale", 4),
        num_heads=cvge_cfg.get("num_heads", 8),
        dropout=cvge_cfg.get("dropout", 0.1),
    )
    # CVGE 는 fp32 로 둔다. 백본은 bf16 이지만 옵티마이저가 잡는 건 CVGE 뿐이고,
    # bf16 마스터 가중치로 AdamW 를 돌리면 lr 1e-4 에서 갱신이 뭉개진다.
    # bf16=true 의 autocast 가 forward 에서 알아서 캐스팅한다.
    cvge.attach(model.model.language_model.layers)
    # Trainer 가 옵티마이저를 만들 때 찾을 수 있도록 모델에 매단다.
    model.cvge = cvge

    n_train = sum(p.numel() for p in cvge.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"CVGE 학습 파라미터 {n_train/1e6:.1f}M / 전체 {n_total/1e9:.2f}B "
          f"({100*n_train/n_total:.2f}%)")

    image_token_id = model.config.image_token_id
    vggt = None                                        # 첫 배치에서 올린다

    debug_nan = cfg.pop("debug_nan", False)

    class CvgeTrainer(Trainer):
        _nan_reported = False

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            nonlocal vggt
            imgs = inputs.pop("vggt_images")
            dev = next(model.parameters()).device
            if vggt is None:
                vggt = load_vggt(vggt_weights, device=dev, dtype=torch.bfloat16)
            f3d = vggt_features(vggt, imgs.to(device=dev, dtype=torch.bfloat16))

            img_mask = inputs["input_ids"] == image_token_id
            core = model.module if hasattr(model, "module") else model

            if debug_nan and not self._nan_reported:
                rank = int(os.environ.get("RANK", 0))
                n_img = img_mask.sum(1).tolist()
                n_lbl = (inputs["labels"] != -100).sum(1).tolist()
                bad = []
                if len(set(n_img)) > 1:
                    bad.append(f"이미지토큰 수 불일치 {n_img}")
                if any(v == 0 for v in n_lbl):
                    bad.append(f"학습 대상 토큰 0개 {n_lbl}")
                if not torch.isfinite(f3d).all():
                    bad.append("f3d 에 nan/inf")
                if bad:
                    self._nan_reported = True
                    print(f"[rank{rank}] 입력 이상: {'; '.join(bad)}", flush=True)

            core.cvge.set_batch(f3d, img_mask)
            try:
                out = model(**inputs)
            finally:
                core.cvge.clear_batch()

            if debug_nan and not self._nan_reported and not torch.isfinite(out.loss):
                self._nan_reported = True
                rank = int(os.environ.get("RANK", 0))
                print(f"[rank{rank}] loss 가 유한하지 않음: {out.loss.item()} "
                      f"이미지토큰={img_mask.sum(1).tolist()} "
                      f"대상토큰={(inputs['labels'] != -100).sum(1).tolist()} "
                      f"f3d유한={bool(torch.isfinite(f3d).all())}", flush=True)
            return (out.loss, out) if return_outputs else out.loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            # 비유한 그래디언트를 clip 전에 버린다. bf16 은 GradScaler 가 없어서
            # 아무도 이걸 걸러주지 않는데, inf 가 clip_grad_norm_ 에 들어가면
            # 계수가 1.0/inf = 0 이 되고 inf * 0 = nan 으로 전 파라미터가 죽는다.
            # 해당 마이크로배치 기여분만 버리고 나머지 누적분은 살린다.
            n_bad = 0
            for p_ in self.model.parameters():
                if p_.grad is not None and not torch.isfinite(p_.grad).all():
                    p_.grad.zero_()
                    n_bad += 1
            if n_bad and not getattr(self, "_bad_grad_warned", False):
                self._bad_grad_warned = True
                print(f"[step {self.state.global_step}] 비유한 그래디언트 {n_bad}개 "
                      f"파라미터에서 발견, 해당 기여분 폐기", flush=True)
            return loss

    train_ds = DlmmCvgeDataset(train_file, max_train_rows)
    eval_ds = DlmmCvgeDataset(eval_file, max_eval_rows) if eval_file else None
    print(f"train rows: {len(train_ds)}"
          + (f"  eval rows: {len(eval_ds)}" if eval_ds else ""))

    targs = TrainingArguments(**cfg)
    trainer = CvgeTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=Collator(processor, max_length),
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))

    # CVGE 만 저장한다. 얼린 7B 를 매번 복제할 이유가 없다.
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
