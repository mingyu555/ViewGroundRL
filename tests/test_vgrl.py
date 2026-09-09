"""Unit tests for the view-grounding pieces. CPU only, no model download.

    python -m pytest tests/test_vgrl.py -q
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgrl.config import ViewGroundingConfig
from vgrl.losses import (
    masked_sequence_mean,
    per_token_kl_k3,
    pick_mask_views,
    view_grounding_loss,
)
from vgrl.masking import ViewMasker
from vgrl.views import CAMERAS, complement_views, evidence_views, parse_views


# ------------------------------------------------------------------ view parsing

def test_parse_object_tags():
    a = ("There is a brown SUV to the back of the ego vehicle and a green light to the "
         "front. The IDs are <c1,CAM_BACK,1088.3,497.5> and <c2,CAM_FRONT,102.0,468.0>.")
    assert parse_views(a) == {"CAM_BACK", "CAM_FRONT"}


def test_evidence_views_sources():
    q = "What is the status of <c1,CAM_FRONT_LEFT,10.0,20.0>?"
    a = "It is stationary, and so is <c2,CAM_BACK_RIGHT,30.0,40.0>."
    assert evidence_views(q, a, source="question") == [CAMERAS.index("CAM_FRONT_LEFT")]
    assert evidence_views(q, a, source="answer") == [CAMERAS.index("CAM_BACK_RIGHT")]
    assert evidence_views(q, a, source="both") == sorted(
        [CAMERAS.index("CAM_FRONT_LEFT"), CAMERAS.index("CAM_BACK_RIGHT")]
    )


def test_no_tags_gives_no_evidence():
    assert evidence_views("What should the ego do?", "Slow down gently.") == []


def test_complement():
    assert complement_views([0, 3]) == [1, 2, 4, 5]


# ------------------------------------------------------------------ view picking

def test_pick_matched_sizes():
    rng = random.Random(0)
    ev, ctrl, ok = pick_mask_views([0, 3], num_views=6, num_to_mask=1, rng=rng)
    assert ok and len(ev) == len(ctrl) == 1
    assert ev[0] in {0, 3} and ctrl[0] not in {0, 3}


def test_pick_invalid_when_no_evidence():
    _, _, ok = pick_mask_views([], num_views=6, num_to_mask=1)
    assert not ok


def test_pick_invalid_when_all_views_are_evidence():
    _, _, ok = pick_mask_views(list(range(6)), num_views=6, num_to_mask=1)
    assert not ok


def test_pick_shrinks_to_available():
    ev, ctrl, ok = pick_mask_views([1], num_views=6, num_to_mask=3)
    assert ok and len(ev) == len(ctrl) == 1  # matched, capped by the smaller side


# ------------------------------------------------------------------ masking

class _FakeImageProcessor:
    patch_size = 14
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __call__(self, images=None, return_tensors=None):
        # a black image under these constants normalises to -1 everywhere
        return {"pixel_values": torch.full((4, 12), -1.0)}


class _FakeProcessor:
    image_processor = _FakeImageProcessor()


def _fake_batch():
    """2 samples x 3 views; view v of sample i has (v+1) patch rows filled with a
    recognisable value so we can assert exactly which rows got blanked."""
    grid = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 3]] * 2)  # rows: 1,2,3,1,2,3
    num_images = [3, 3]
    rows = []
    for sample in range(2):
        for view in range(3):
            n = view + 1
            rows.append(torch.full((n, 12), float(sample * 10 + view)))
    return torch.cat(rows, dim=0), grid, num_images


def test_row_spans_align_with_grid():
    pv, grid, num_images = _fake_batch()
    spans = ViewMasker.row_spans(grid, num_images)
    assert spans == [[(0, 1), (1, 3), (3, 6)], [(6, 7), (7, 9), (9, 12)]]
    assert spans[-1][-1][1] == pv.size(0)


def test_mask_only_touches_requested_view():
    pv, grid, num_images = _fake_batch()
    masker = ViewMasker(_FakeProcessor(), mode="black")
    out = masker.mask(pv, grid, num_images, [[1], []])

    # sample 0 view 1 -> rows 1:3 blanked
    assert torch.allclose(out[1:3], torch.full((2, 12), -1.0))
    # everything else untouched
    untouched = torch.cat([out[0:1], out[3:]], dim=0)
    expected = torch.cat([pv[0:1], pv[3:]], dim=0)
    assert torch.allclose(untouched, expected)
    # input not mutated
    assert torch.allclose(pv[1:3], torch.full((2, 12), 1.0))


def test_mask_multiple_samples_and_views():
    pv, grid, num_images = _fake_batch()
    masker = ViewMasker(_FakeProcessor(), mode="zero")
    out = masker.mask(pv, grid, num_images, [[0, 2], [1]])
    assert torch.allclose(out[0:1], torch.zeros(1, 12))    # s0 v0
    assert torch.allclose(out[3:6], torch.zeros(3, 12))    # s0 v2
    assert torch.allclose(out[7:9], torch.zeros(2, 12))    # s1 v1
    assert torch.allclose(out[1:3], pv[1:3])               # s0 v1 kept
    assert torch.allclose(out[6:7], pv[6:7])               # s1 v0 kept


def test_mask_rejects_out_of_range_view():
    pv, grid, num_images = _fake_batch()
    masker = ViewMasker(_FakeProcessor())
    with pytest.raises(IndexError):
        masker.mask(pv, grid, num_images, [[5], []])


# ------------------------------------------------------------------ KL + loss

def test_k3_is_zero_for_identical_distributions():
    lp = torch.log(torch.tensor([0.3, 0.5, 0.2]))
    assert torch.allclose(per_token_kl_k3(lp, lp), torch.zeros(3), atol=1e-7)


def test_k3_is_non_negative():
    torch.manual_seed(0)
    p = torch.log_softmax(torch.randn(200), dim=-1)
    q = torch.log_softmax(torch.randn(200), dim=-1)
    assert (per_token_kl_k3(p, q) >= -1e-6).all()


def test_k3_estimator_approximates_true_kl():
    """Averaged over samples from p, k3 should match the analytic KL(p||q)."""
    torch.manual_seed(0)
    logits_p, logits_q = torch.randn(50), torch.randn(50)
    logp = torch.log_softmax(logits_p, -1)
    logq = torch.log_softmax(logits_q, -1)
    true_kl = (logp.exp() * (logp - logq)).sum()

    idx = torch.multinomial(logp.exp(), 200_000, replacement=True)
    est = per_token_kl_k3(logp[idx], logq[idx]).mean()
    assert torch.allclose(est, true_kl, rtol=0.05), f"{est} vs {true_kl}"


def test_masked_sequence_mean_ignores_padding():
    per_token = torch.tensor([[1.0, 3.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert torch.allclose(masked_sequence_mean(per_token, mask), torch.tensor([2.0]))


def _logps(b=2, t=4, offset=0.0):
    torch.manual_seed(1)
    return torch.log_softmax(torch.randn(b, t, 7) + offset, dim=-1).max(-1).values


def test_sign_grounded_model_gets_lower_loss():
    """A grounded model (large kl1, small kl2) must incur a *smaller* loss than an
    anti-grounded one under sign='maximize_margin'."""
    mask = torch.ones(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    orig = torch.full((2, 4), -1.0)

    grounded = view_grounding_loss(
        orig, torch.full((2, 4), -4.0), torch.full((2, 4), -1.01),
        mask, valid, coef=0.1,
    )
    anti = view_grounding_loss(
        orig, torch.full((2, 4), -1.01), torch.full((2, 4), -4.0),
        mask, valid, coef=0.1,
    )
    assert grounded.margin.mean() > 0 > anti.margin.mean()
    assert grounded.loss < anti.loss


def test_literal_sign_is_the_opposite():
    mask = torch.ones(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    orig = torch.full((2, 4), -1.0)
    kw = dict(completion_mask=mask, valid=valid, coef=0.1)
    a = view_grounding_loss(orig, torch.full((2, 4), -4.0), torch.full((2, 4), -1.01),
                            sign="maximize_margin", **kw)
    b = view_grounding_loss(orig, torch.full((2, 4), -4.0), torch.full((2, 4), -1.01),
                            sign="literal", **kw)
    assert torch.allclose(a.loss, -b.loss)


def test_kl1_only_ignores_control_branch():
    mask = torch.ones(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    out = view_grounding_loss(
        torch.full((2, 4), -1.0), torch.full((2, 4), -3.0), torch.full((2, 4), -9.0),
        mask, valid, coef=1.0, sign="kl1_only",
    )
    assert torch.allclose(out.kl2, torch.zeros(2))
    assert torch.allclose(out.margin, out.kl1)


def test_invalid_samples_are_excluded():
    mask = torch.ones(2, 4)
    valid = torch.tensor([True, False])
    out = view_grounding_loss(
        torch.full((2, 4), -1.0),
        torch.stack([torch.full((4,), -4.0), torch.full((4,), -100.0)]),
        torch.full((2, 4), -1.01),
        mask, valid, coef=1.0, margin_clip=None,
    )
    assert out.num_valid == 1
    # the excluded row's enormous kl1 must not leak into the loss
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, -out.margin[0])


def test_all_invalid_gives_zero_loss_that_still_backprops():
    mask = torch.ones(2, 4)
    valid = torch.zeros(2, dtype=torch.bool)
    orig = torch.full((2, 4), -1.0, requires_grad=True)
    out = view_grounding_loss(orig, torch.full((2, 4), -4.0), torch.full((2, 4), -1.0),
                              mask, valid, coef=1.0)
    assert out.loss.item() == 0.0
    out.loss.backward()  # must not raise: keeps the graph connected
    assert orig.grad is not None


def test_margin_clip_bounds_the_term():
    mask = torch.ones(1, 4)
    valid = torch.ones(1, dtype=torch.bool)
    out = view_grounding_loss(
        torch.full((1, 4), -1.0), torch.full((1, 4), -50.0), torch.full((1, 4), -1.0),
        mask, valid, coef=1.0, margin_clip=5.0,
    )
    assert out.margin.abs().max() <= 5.0
    assert abs(out.loss.item()) <= 5.0


def test_loss_is_differentiable_through_both_branches():
    mask = torch.ones(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    orig = torch.full((2, 4), -1.0, requires_grad=True)
    ev = torch.full((2, 4), -2.0, requires_grad=True)
    ctrl = torch.full((2, 4), -1.5, requires_grad=True)
    view_grounding_loss(orig, ev, ctrl, mask, valid, coef=0.1).loss.backward()
    for t in (orig, ev, ctrl):
        assert t.grad is not None and torch.isfinite(t.grad).all()


# ------------------------------------------------- k1 (방향성) 추정기

def _flat(v, shape=(2, 4)):
    return torch.full(shape, v)


def test_k1_is_negative_when_masking_makes_the_answer_easier():
    """k3 가 없는 접지를 보고하는 그 경우를 k1 은 음수로 잡는다.

    학습 전 Qwen2.5-VL 이 NuInstruct 에서 정확히 이랬다: k3 kl1 0.839 로 접지가
    있어 보였지만, 정답 로그확률은 오히려 올라가 k1 은 -0.030 이었다.
    """
    orig, ctrl = _flat(-1.0), _flat(-1.02)
    easier = _flat(-0.5)          # 근거뷰를 가리자 정답이 더 쉬워졌다
    k3 = view_grounding_loss(orig, easier, ctrl, torch.ones(2, 4),
                             torch.ones(2, dtype=torch.bool), estimator="k3")
    k1 = view_grounding_loss(orig, easier, ctrl, torch.ones(2, 4),
                             torch.ones(2, dtype=torch.bool), estimator="k1")
    assert k3.kl1[0] > 0          # 부호를 버려서 접지가 있다고 보고한다
    assert k1.kl1[0] < 0          # 방향성 추정기는 없다고 본다


def test_k1_margin_cancels_the_unmasked_branch():
    """k1 에서 margin = kl1-kl2 = logp(대조뷰 가림) - logp(근거뷰 가림).

    전체뷰 항이 정확히 소거되므로 관점항은 두 마스크 분기 사이의 대조 손실이 된다.
    관점항만 놓고 보면 전체뷰 순전파가 필요하지 않다는 뜻이다.
    """
    orig, ev, ctrl = _flat(-1.0), _flat(-3.0), _flat(-1.02)
    out = view_grounding_loss(orig, ev, ctrl, torch.ones(2, 4),
                              torch.ones(2, dtype=torch.bool), estimator="k1")
    assert out.margin[0] == pytest.approx(-1.02 - (-3.0), abs=1e-5)
    # 전체뷰 분기를 바꿔도 margin 이 안 움직인다
    shifted = view_grounding_loss(_flat(-7.0), ev, ctrl, torch.ones(2, 4),
                                  torch.ones(2, dtype=torch.bool), estimator="k1")
    assert shifted.margin[0] == pytest.approx(out.margin[0], abs=1e-5)


def test_k1_clip_is_two_sided():
    """k1 은 음수가 정상값이라 kl_clip 이 위아래를 함께 잘라야 한다."""
    orig, ev, ctrl = _flat(-1.0), _flat(-40.0), _flat(40.0)
    out = view_grounding_loss(orig, ev, ctrl, torch.ones(2, 4),
                              torch.ones(2, dtype=torch.bool),
                              estimator="k1", kl_clip=5.0)
    assert out.kl1[0] == pytest.approx(5.0)
    assert out.kl2[0] == pytest.approx(-5.0)


def test_config_rejects_bad_estimator():
    with pytest.raises(ValueError):
        ViewGroundingConfig(estimator="k2").validate()


# ------------------------------------------------------------------ config

def test_config_rejects_bad_sign():
    with pytest.raises(ValueError):
        ViewGroundingConfig(sign="nope").validate()


def test_config_rejects_unmatched_mask_budget():
    # masking 4 of 6 views leaves only 2 for the control set -> no matched split
    with pytest.raises(ValueError):
        ViewGroundingConfig(num_views_to_mask=4).validate()


def test_config_defaults_are_valid():
    ViewGroundingConfig().validate()
