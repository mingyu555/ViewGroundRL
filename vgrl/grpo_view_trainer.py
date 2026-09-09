"""GRPO with a view-grounding term.

`ViewGroundedGRPOTrainer` subclasses TRL's `GRPOTrainer` and adds
`-coef * (kl1 - kl2)` to the loss, where the two KLs come from re-scoring the
already-sampled completion with one camera view blanked (see vgrl/losses.py).

Three integration points, all narrow on purpose so the base trainer keeps owning
the GRPO maths:

1. `_set_signature_columns_if_needed` keeps our `evidence_views` dataset column
   from being dropped by `remove_unused_columns`.
2. `_generate_and_score_completions` attaches the per-sample evidence views to
   the batch dict. TRL's `shuffle_sequence_dict` / `split_tensor_dict` carry
   plain lists along the batch axis, so the field stays aligned with the
   completions through shuffling and gradient-accumulation splitting.
3. `_get_per_token_logps_and_entropies` is wrapped so the policy log-probs the
   base loss already computed can be reused for kl1/kl2 instead of paying for a
   third forward pass.

Pinned against TRL 1.9.x; `_assert_trl_api` fails loudly rather than silently
skipping the term if those internals move.
"""

from __future__ import annotations

import inspect
import hashlib
import random
from collections import defaultdict

import torch
from trl import GRPOTrainer

from .config import ViewGroundingConfig
from .losses import pick_mask_views, view_grounding_loss
from .tokens import content_token_mask
from .masking import ViewMasker
from .views import evidence_views as parse_evidence_views

EVIDENCE_COLUMN = "evidence_views"
_MASK_KEY = "_vg_mask_flags"
_BATCH_KEY = "vg_evidence_views"


def _assert_trl_api() -> None:
    """Fail early if the base-class internals this trainer hooks have changed."""
    sig = inspect.signature(GRPOTrainer._get_per_token_logps_and_entropies)
    for name in ("pixel_values", "image_grid_thw", "num_images", "logits_to_keep"):
        if name not in sig.parameters:
            raise RuntimeError(
                "This trainer targets TRL 1.9.x: "
                f"GRPOTrainer._get_per_token_logps_and_entropies has no {name!r} parameter. "
                "Re-check the integration points in vgrl/grpo_view_trainer.py."
            )
    for meth in ("_compute_loss", "_generate_and_score_completions"):
        if not hasattr(GRPOTrainer, meth):
            raise RuntimeError(f"TRL GRPOTrainer is missing {meth}; unsupported version.")


class ViewGroundedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, vg_config: ViewGroundingConfig | None = None, **kwargs):
        _assert_trl_api()
        super().__init__(*args, **kwargs)

        self.vg_config = vg_config or ViewGroundingConfig()
        self.vg_config.validate()
        self._vg_rng = random.Random(self.vg_config.seed)
        self._vg_masker = ViewMasker(
            self.processing_class,
            mode=self.vg_config.mask_mode,
            noise_std=self.vg_config.noise_std,
        )
        self._vg_metrics: dict[str, list[float]] = defaultdict(list)

        # set while the base loss runs, so we can grab the policy log-probs
        self._vg_capture = False
        self._vg_captured_logps: torch.Tensor | None = None

    # ------------------------------------------------------------------ plumbing

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if EVIDENCE_COLUMN not in self._signature_columns:
            self._signature_columns.append(EVIDENCE_COLUMN)

    @staticmethod
    def _sample_evidence(sample: dict, cfg: ViewGroundingConfig) -> list[int]:
        """Evidence view indices for one dataset row.

        Prefers a precomputed `evidence_views` column; falls back to parsing the
        DriveLM object tags out of the prompt/answer text so the trainer still
        works on datasets that were not pre-annotated.
        """
        ev = sample.get(EVIDENCE_COLUMN)
        if ev is not None:
            return [int(v) for v in ev]

        prompt = sample.get("prompt")
        if isinstance(prompt, list):  # conversational
            question = " ".join(
                part.get("text", "")
                for msg in prompt
                for part in (
                    msg["content"] if isinstance(msg.get("content"), list) else [{"text": msg.get("content", "")}]
                )
                if isinstance(part, dict)
            )
        else:
            question = prompt or ""
        answer = sample.get("solution") or sample.get("answer") or ""
        return parse_evidence_views(question, answer, source=cfg.answer_view_source)

    # ------------------------------------------------------------------
    # 롤아웃 혼합 (mode="rollout_mask")
    #
    # 측정으로 확인한 실패 모드: RL 은 drop_ev 를 거의 안 건드리고(3B +0.0014)
    # drop_ctrl 을 올려서(3B +0.0096, 7B풀FT +0.0100) gap 을 깎는다. 즉 모델이
    # '아무 뷰든 가리면 흔들리는' 방향으로 간다.
    #
    # 그래서 KL 항으로 '근거뷰 의존'을 보상하는 대신, 그룹의 일부 롤아웃을
    # '대조뷰 셀을 가린' 입력에서 뽑는다. 보상 함수는 그대로다. 가린 조건에서
    # 맞힌 답은 양의 어드밴티지, 틀린 답은 음의 어드밴티지를 받으므로
    # '무관한 뷰가 빠져도 맞혀라'가 직접 학습된다. 추가 생성이 없어 비용은
    # plain GRPO 와 같다.
    # ------------------------------------------------------------------
    def _mask_grid_cell(self, img, cell: int):
        """합성 격자(2x3)에서 셀 하나를 검게. DriveLMM-o1 은 6뷰를 한 장으로
        합쳐 넣으므로 '뷰 가림' = '셀 가림' 이다."""
        from PIL import ImageDraw

        cfg = self.vg_config
        cols = cfg.grid_cols
        rows = max(1, cfg.grid_cells // cols)
        w, h = img.size
        cw, ch = w // cols, h // rows
        out = img.copy()
        x0, y0 = (cell % cols) * cw, (cell // cols) * ch
        ImageDraw.Draw(out).rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=(0, 0, 0))
        return out

    def _pick_control_cell(self, evidence, key: str):
        cand = [i for i in range(self.vg_config.grid_cells) if i not in evidence]
        if not cand:
            return None
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        return cand[h % len(cand)]

    def _apply_rollout_mask(self, generation_batch, evidence):
        """각 프롬프트 그룹의 마지막 N 개 행의 이미지를 대조뷰 가림으로 바꾼다.

        TRL 의 RepeatSampler 가 프롬프트를 num_generations 번 연속으로 내보내므로
        인덱스 // num_generations 가 그룹 번호, % 가 그룹 내 위치다."""
        cfg = self.vg_config
        G = self.args.num_generations
        n_mask = min(cfg.masked_rollouts, G - 1)
        flags = []
        for i, sample in enumerate(generation_batch):
            masked = (i % G) >= (G - n_mask)
            ev = evidence[i] or []
            cell = self._pick_control_cell(ev, str(sample.get("id", i))) if masked else None
            if masked and cell is not None and sample.get("images"):
                imgs = list(sample["images"])
                imgs[0] = self._mask_grid_cell(imgs[0], cell)
                sample["images"] = imgs
            else:
                masked = False
            flags.append(masked)
        return flags

    def _generate_and_score_completions(self, generation_batch):
        evidence = [self._sample_evidence(s, self.vg_config) for s in generation_batch]
        mask_flags = None
        if self.vg_config.enabled and self.vg_config.mode == "rollout_mask":
            mask_flags = self._apply_rollout_mask(generation_batch, evidence)
        output = super()._generate_and_score_completions(generation_batch)
        if mask_flags is not None:
            output[_MASK_KEY] = mask_flags
            self._renormalize_per_condition(output, mask_flags)
        # `output` rows are 1:1 with `generation_batch` rows: TRL's RepeatSampler
        # already emits each prompt num_generations times, so no re-expansion here.
        if len(evidence) != len(output["completion_ids"]):
            raise RuntimeError(
                f"evidence rows ({len(evidence)}) != completion rows "
                f"({len(output['completion_ids'])}); batch alignment assumption broken."
            )
        output[_BATCH_KEY] = evidence
        return output

    def _renormalize_per_condition(self, output, mask_flags):
        """조건별 어드밴티지 정규화.

        두 조건을 한 그룹으로 묶으면 가린 롤아웃이 그룹 평균을 낮춰 전체뷰
        롤아웃의 어드밴티지가 부풀려진다 (8개 중 2개를 가리면 편향이 작지 않다).
        조건 안에서만 평균·표준편차를 다시 잡는다."""
        if not self.vg_config.normalize_per_condition:
            return
        adv = output.get("advantages")
        if adv is None:
            return
        G = self.args.num_generations
        flags = torch.tensor(mask_flags, device=adv.device, dtype=torch.bool)
        n_groups = adv.numel() // G
        a = adv.view(n_groups, G)
        f = flags.view(n_groups, G)
        for cond in (False, True):
            sel = (f == cond)
            cnt = sel.sum(dim=1, keepdim=True)
            if not sel.any():
                continue
            vals = torch.where(sel, a, torch.zeros_like(a))
            mean = vals.sum(dim=1, keepdim=True) / cnt.clamp(min=1)
            var = torch.where(sel, (a - mean) ** 2, torch.zeros_like(a))
            std = (var.sum(dim=1, keepdim=True) / cnt.clamp(min=1)).sqrt()
            newa = (a - mean) / (std + 1e-4)
            a = torch.where(sel & (cnt > 1), newa, a)
        output["advantages"] = a.view(-1)
        self._vg_metrics["mask_frac"].append(float(flags.float().mean()))

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        out = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if self._vg_capture and self._vg_captured_logps is None:
            self._vg_captured_logps = out[0]
            self._vg_capture = False
        return out

    # ------------------------------------------------------------------ the term

    def _vg_should_apply(self, inputs) -> bool:
        if not self.vg_config.enabled:
            return False
        # rollout_mask 모드는 손실항을 쓰지 않는다. 마스킹은 생성 단계에서 끝나고
        # 학습은 plain GRPO 그대로다 (그래서 추가 forward 가 없다).
        if self.vg_config.mode == "rollout_mask":
            return False
        if self.state.global_step < self.vg_config.warmup_steps:
            return False
        return (
            inputs.get("pixel_values") is not None
            and inputs.get("image_grid_thw") is not None
            and inputs.get("num_images") is not None
            and inputs.get(_BATCH_KEY) is not None
        )

    def _vg_logps(self, model, inputs, pixel_values) -> torch.Tensor:
        """Per-token log-probs of the sampled completion under given pixel values."""
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logps, _, _ = super()._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            inputs["completion_ids"].size(1),
            compute_entropy=False,
            pixel_values=pixel_values,
            image_grid_thw=inputs["image_grid_thw"],
            num_images=inputs["num_images"],
        )
        return logps

    def _view_grounding_term(self, model, inputs, logps_orig) -> torch.Tensor:
        cfg = self.vg_config
        num_images = list(inputs["num_images"])
        evidence_rows = inputs[_BATCH_KEY]

        ev_picks, ctrl_picks, valid_flags = [], [], []
        for i, ev in enumerate(evidence_rows):
            # 합성 격자 입력은 이미지가 1 장이므로 num_images 로 뷰 수를 세면
            # n_views=1 이 되어 근거/대조 분할이 항상 실패한다 (실측: num_valid 0).
            # 이때 뷰 수는 격자 셀 수다.
            n_views = (cfg.grid_cells if cfg.grid_input
                       else min(cfg.num_views, num_images[i]))
            e, c, ok = pick_mask_views(
                list(ev), n_views, cfg.num_views_to_mask, rng=self._vg_rng
            )
            ev_picks.append(e)
            ctrl_picks.append(c)
            valid_flags.append(ok)

        valid = torch.tensor(valid_flags, device=logps_orig.device, dtype=torch.bool)

        # Whether to run the masked branches must be decided *identically on every
        # rank*. The branches drive collectives (DDP reductions inside the extra
        # forward passes), so a rank that skips them enqueues fewer NCCL ops than
        # its peers and the process group deadlocks — observed as a collective
        # timeout with one rank ~150 works behind the others.
        #
        # This never fired on DriveLM because that RL set was filtered to
        # vg_usable=100%. On nuScenes only 51% of frames can form a matched
        # evidence/control split, so a micro-batch of 4 is all-unusable 5.6% of the
        # time and at least one of 4 ranks diverges on 20.5% of steps — it cannot
        # survive 200 steps. Agreeing here costs one bool all-reduce per step.
        any_valid = valid.any()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = any_valid.to(torch.uint8)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            any_valid = flag.bool()

        if not bool(any_valid):
            self._vg_metrics["num_valid"].append(0.0)
            return logps_orig.sum() * 0.0

        pv = inputs["pixel_values"]
        grid = inputs["image_grid_thw"]

        # 합성 격자(2x3 한 장) 입력이면 이미지 단위가 아니라 셀 단위로 지운다.
        _mask = (self._vg_masker.mask_grid_cells
                 if self.vg_config.grid_input else self._vg_masker.mask)
        _kw = ({"grid_rows": self.vg_config.grid_cells // self.vg_config.grid_cols,
                "grid_cols": self.vg_config.grid_cols}
               if self.vg_config.grid_input else {})
        pv_evidence = _mask(pv, grid, num_images, ev_picks, **_kw)
        logps_evidence = self._vg_logps(model, inputs, pv_evidence)

        if cfg.sign == "kl1_only":
            logps_control = None
        else:
            pv_control = _mask(pv, grid, num_images, ctrl_picks, **_kw)
            if cfg.detach_control_branch:
                with torch.no_grad():
                    logps_control = self._vg_logps(model, inputs, pv_control)
            else:
                logps_control = self._vg_logps(model, inputs, pv_control)

        mask = inputs["completion_mask"]
        if "tool_mask" in inputs:
            mask = mask * inputs["tool_mask"]
        if cfg.token_filter == "content":
            cm = content_token_mask(inputs["completion_ids"],
                                    self.processing_class.tokenizer)
            mask = mask * cm.to(mask.dtype)
            # 내용어가 하나도 없는 롤아웃은 margin 이 0 이 되고, softplus 는 그 0 에
            # 기울기 0.5 를 걸어 아무것도 아닌 것에 압력을 준다. 무효 처리한다.
            valid = valid & (mask.sum(dim=-1) > 0)

        out = view_grounding_loss(
            logps_orig=logps_orig,
            logps_mask_evidence=logps_evidence,
            logps_mask_control=logps_control,
            completion_mask=mask,
            valid=valid,
            coef=cfg.coef,
            sign=cfg.sign,
            margin_clip=cfg.margin_clip,
            kl_clip=cfg.kl_clip,
            estimator=cfg.estimator,
            transform=cfg.transform,
        )

        if out.num_valid:
            vf = valid.to(out.kl1.dtype)
            denom = vf.sum()
            self._vg_metrics["kl1"].append(((out.kl1 * vf).sum() / denom).item())
            self._vg_metrics["kl2"].append(((out.kl2 * vf).sum() / denom).item())
            self._vg_metrics["margin"].append(((out.margin * vf).sum() / denom).item())
            self._vg_metrics["loss"].append(out.loss.item())
        self._vg_metrics["num_valid"].append(float(out.num_valid))
        self._vg_metrics["frac_valid"].append(out.num_valid / max(len(valid_flags), 1))
        return out.loss

    def _compute_loss(self, model, inputs):
        apply_vg = self._vg_should_apply(inputs)
        self._vg_capture = apply_vg
        self._vg_captured_logps = None

        loss = super()._compute_loss(model, inputs)

        if apply_vg:
            if self._vg_captured_logps is None:
                raise RuntimeError(
                    "policy log-probs were not captured from the base loss; the TRL "
                    "internals this trainer hooks have changed."
                )
            loss = loss + self._view_grounding_term(model, inputs, self._vg_captured_logps)

        self._vg_capture = False
        self._vg_captured_logps = None
        return loss

    # ------------------------------------------------------------------ logging

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        prefix = self.vg_config.log_prefix
        for name, values in self._vg_metrics.items():
            if values:
                logs[f"{prefix}/{name}"] = sum(values) / len(values)
        self._vg_metrics.clear()
        super().log(logs, start_time)
