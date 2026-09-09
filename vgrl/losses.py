"""View-grounding loss: reward the policy for actually depending on the view
that carries the evidence, and for being indifferent to views that do not.

Two counterfactual forward passes are run over the *same* sampled completion:

  kl1 = KL( p(.|all views)  ||  p(.|evidence view blanked) )
  kl2 = KL( p(.|all views)  ||  p(.|a non-evidence view blanked) )

and the contrast

  margin = kl1 - kl2

is the quantity of interest. A model that reads the right view has a large kl1
(blanking the evidence changes its answer) and a small kl2 (blanking an
irrelevant view does not), so `margin` should be *large*.

SIGN. The loss returned is `-coef * margin`, because the trainer *minimises* what
it is given. Adding `+coef * margin` to a minimised objective would train the
model to become insensitive to the evidence view and sensitive to irrelevant
ones — the exact opposite of the intent. `sign="literal"` is available for
ablations, and `sign="kl1_only"` reduces this to plain PAPO (no control branch).

Why kl2 at all: kl1 on its own is maximised just as well by becoming globally
brittle to *any* missing pixels. Subtracting kl2 removes that shortcut, since a
uniformly brittle model raises kl2 alongside kl1 and gains nothing. For the
contrast to isolate *which* view matters rather than *how much* was removed, both
branches must blank the same number of views — enforced in `pick_mask_views`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


def per_token_kl_k3(
    logps_p: torch.Tensor,
    logps_q: torch.Tensor,
) -> torch.Tensor:
    """Low-variance positive estimator of KL(p || q) per token.

    The completions were sampled from p, so with r = q(x)/p(x) the k3 estimator
    `r - log r - 1` is non-negative and unbiased for KL(p || q). Same estimator
    TRL/GRPO uses for the reference-model KL, and the one PAPO uses for its
    implicit perception loss.
    """
    delta = logps_q - logps_p
    return torch.exp(delta) - delta - 1.0


def per_token_kl_k1(
    logps_p: torch.Tensor,
    logps_q: torch.Tensor,
) -> torch.Tensor:
    """Directional per-token estimator: -log r = log p - log q.

    k3 is non-negative by construction, so it scores "the distribution moved"
    rather than "the correct answer got harder". Measured on NuInstruct, the
    untrained model gets k3 kl1/kl2 = 0.839/0.255 (a 3.3x margin) while its k1
    values are -0.030/-0.029 — no selectivity at all. k3 reported grounding that
    is not there, because removing the evidence view moves the distribution
    without moving it toward the answer. Here the sign carries the whole meaning,
    so k3's variance reduction costs more than it saves.
    """
    return logps_p - logps_q


def masked_sequence_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid completion tokens, per sequence -> shape (B,)."""
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (per_token * mask).sum(dim=-1) / denom


@dataclass
class ViewGroundingOutput:
    loss: torch.Tensor          # scalar, already signed for minimisation
    kl1: torch.Tensor           # (B,) per-sequence, detached
    kl2: torch.Tensor           # (B,)
    margin: torch.Tensor        # (B,)
    valid: torch.Tensor         # (B,) bool — sample contributed to the loss
    num_valid: int


def view_grounding_loss(
    logps_orig: torch.Tensor,
    logps_mask_evidence: torch.Tensor,
    logps_mask_control: torch.Tensor | None,
    completion_mask: torch.Tensor,
    valid: torch.Tensor,
    coef: float = 0.02,
    sign: str = "maximize_margin",
    margin_clip: float | None = 10.0,
    kl_clip: float | None = None,
    estimator: str = "k3",
    transform: str = "linear",
) -> ViewGroundingOutput:
    """Combine the three branches into a scalar loss.

    Args:
        logps_orig: (B, T) per-token log-probs of the sampled completion under
            the unmasked images.
        logps_mask_evidence: (B, T) same completion, evidence view(s) blanked.
        logps_mask_control: (B, T) same completion, non-evidence view(s) blanked.
            `None` collapses to PAPO (kl2 = 0).
        completion_mask: (B, T) 1 for real completion tokens.
        valid: (B,) bool — False for samples with no usable evidence/control view
            split (e.g. a question that names no camera, or one that names all
            six, leaving no control view). Those samples contribute 0.
        coef: weight of the term. PAPO-GRPO-3B uses 0.02 for its analogous
            gamma and ablates 0.005-0.04 (collapse at 0.04); see config.py.
        sign: "maximize_margin" (default), "literal", or "kl1_only".
        margin_clip: cap |margin| per sequence before weighting. Guards against
            the KL-hacking failure mode PAPO reports, where the model drives the
            divergence up degenerately instead of grounding.
        kl_clip: optional cap applied to kl1 and kl2 individually.
        transform: "linear" (loss = -coef * margin) or "softplus"
            (loss = coef * softplus(-margin)). With the k1 estimator the margin
            is a two-way logit difference — log p(y|control masked) minus
            log p(y|evidence masked) — so softplus(-margin) is the cross-entropy
            of picking the control branch, i.e. the InfoNCE form. Its gradient is
            -sigmoid(-margin), which is bounded by 1 and vanishes as the margin
            grows, so the objective cannot run away the way the linear form did
            (measured on NuInstruct: k3 kl1 reached 3.8e13, directional drop
            +4.64 nats). Near margin 0 it is the linear form with half the
            gradient, so coef should roughly double to match.
        estimator: "k3" (r - log r - 1, non-negative) or "k1" (-log r,
            directional). See `per_token_kl_k1` — "k1" is what the diagnostic
            uses, and the two disagree on whether an untrained model is grounded
            at all. "k3" stays the default so the finished runs remain
            reproducible.
    """
    est = {"k3": per_token_kl_k3, "k1": per_token_kl_k1}.get(estimator)
    if est is None:
        raise ValueError(f"unknown estimator: {estimator!r}")

    kl1_tok = est(logps_orig, logps_mask_evidence)
    kl1 = masked_sequence_mean(kl1_tok, completion_mask)

    if logps_mask_control is None or sign == "kl1_only":
        kl2 = torch.zeros_like(kl1)
    else:
        kl2_tok = est(logps_orig, logps_mask_control)
        kl2 = masked_sequence_mean(kl2_tok, completion_mask)

    if kl_clip is not None:
        # k1 은 음수가 의미 있는 값이라 위쪽만 자르면 분포가 한쪽으로 눌린다.
        lo = -kl_clip if estimator == "k1" else None
        kl1 = kl1.clamp(min=lo, max=kl_clip)
        kl2 = kl2.clamp(min=lo, max=kl_clip)

    margin = kl1 - kl2
    if margin_clip is not None:
        margin = margin.clamp(min=-margin_clip, max=margin_clip)

    valid_f = valid.to(margin.dtype)
    num_valid = int(valid.sum().item())

    if num_valid == 0:
        loss = margin.sum() * 0.0
    else:
        mean_margin = (margin * valid_f).sum() / valid_f.sum()
        if sign in {"maximize_margin", "kl1_only"}:
            if transform == "softplus":
                # 표본별로 감싸고 평균한다. softplus 는 비선형이라
                # mean(softplus(-m_i)) != softplus(-mean(m_i)) 이고, 앞쪽이
                # "표본마다 올바른 분기를 고를 확률"이라는 해석에 맞다.
                per = torch.nn.functional.softplus(-margin)
                loss = coef * (per * valid_f).sum() / valid_f.sum()
            elif transform == "linear":
                loss = -coef * mean_margin
            else:
                raise ValueError(f"unknown transform: {transform!r}")
        elif sign == "literal":
            loss = coef * mean_margin
        else:
            raise ValueError(f"unknown sign: {sign!r}")

    return ViewGroundingOutput(
        loss=loss,
        kl1=kl1.detach(),
        kl2=kl2.detach(),
        margin=margin.detach(),
        valid=valid,
        num_valid=num_valid,
    )


def pick_mask_views(
    evidence: list[int],
    num_views: int,
    num_to_mask: int = 1,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], bool]:
    """Choose matched evidence / control view sets for one sample.

    Returns `(evidence_pick, control_pick, valid)`. Both picks have the same
    length so that `kl1 - kl2` reflects *which* view was removed rather than how
    much of the image was removed. `valid` is False when the sample cannot supply
    that matched split, in which case it is excluded from the loss.
    """
    rng = rng or random
    ev = sorted({v for v in evidence if 0 <= v < num_views})
    ctrl_pool = [v for v in range(num_views) if v not in set(ev)]

    k = min(num_to_mask, len(ev), len(ctrl_pool))
    if k == 0:
        return [], [], False

    return rng.sample(ev, k), rng.sample(ctrl_pool, k), True
