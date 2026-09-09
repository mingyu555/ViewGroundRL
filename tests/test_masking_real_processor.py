"""Integration test: view masking against a real Qwen2.5-VL image processor.

The unit tests in test_vgrl.py use a fake processor, which proves the row
arithmetic but not that the fill value or the patch layout assumptions match a
real Qwen2.5-VL. This test blanks a view with `ViewMasker` and compares the result
against actually feeding a black image through the processor in that slot — they
must be bit-identical.

Needs a local processor directory (preprocessor_config.json + tokenizer); no model
weights. Point VGRL_TEST_PROCESSOR at one, e.g.

    VGRL_TEST_PROCESSOR=/mnt/ssd1/minddriver_repro/ckpt \
        python -m pytest tests/test_masking_real_processor.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSOR = os.environ.get("VGRL_TEST_PROCESSOR")
pytestmark = pytest.mark.skipif(
    not PROCESSOR or not os.path.isdir(PROCESSOR),
    reason="set VGRL_TEST_PROCESSOR to a local Qwen2.5-VL processor directory",
)


@pytest.fixture(scope="module")
def image_processor():
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(PROCESSOR, trust_remote_code=True)
    return proc, proc.image_processor


def _six_views():
    from PIL import Image

    # deliberately different sizes per view, so a bug in the row arithmetic
    # cannot hide behind uniform grids
    return [Image.new("RGB", (448 - 28 * i, 308), (10 * i + 5, 20, 30)) for i in range(6)]


def test_row_spans_cover_every_patch_row(image_processor):
    _, ip = image_processor
    from vgrl.masking import ViewMasker

    out = ip(images=_six_views(), return_tensors="pt")
    spans = ViewMasker.row_spans(out["image_grid_thw"], [6])
    assert spans[0][0][0] == 0
    assert spans[0][-1][1] == out["pixel_values"].size(0)
    for (_, prev_end), (start, _) in zip(spans[0], spans[0][1:]):
        assert prev_end == start, "view spans must be contiguous"


@pytest.mark.parametrize("view", [0, 2, 5])
def test_masked_view_matches_real_black_encoding(image_processor, view):
    import torch

    proc, ip = image_processor
    from PIL import Image

    from vgrl.masking import ViewMasker

    imgs = _six_views()
    out = ip(images=imgs, return_tensors="pt")
    pv, grid = out["pixel_values"], out["image_grid_thw"]

    ours = ViewMasker(proc, mode="black").mask(pv, grid, [6], [[view]])

    reference_imgs = list(imgs)
    reference_imgs[view] = Image.new("RGB", imgs[view].size, (0, 0, 0))
    ref = ip(images=reference_imgs, return_tensors="pt")["pixel_values"]

    start, end = ViewMasker.row_spans(grid, [6])[0][view]
    assert torch.allclose(ours[start:end], ref[start:end], atol=1e-5), (
        "blanking a view must reproduce what the processor emits for a black image"
    )
    kept_ours = torch.cat([ours[:start], ours[end:]])
    kept_orig = torch.cat([pv[:start], pv[end:]])
    assert torch.equal(kept_ours, kept_orig), "other views must be untouched"


def test_multi_sample_batch_spans(image_processor):
    _, ip = image_processor
    from vgrl.masking import ViewMasker

    imgs = _six_views()
    out = ip(images=imgs + imgs, return_tensors="pt")
    spans = ViewMasker.row_spans(out["image_grid_thw"], [6, 6])
    assert len(spans) == 2
    assert spans[1][0][0] == spans[0][-1][1]
    assert spans[1][-1][1] == out["pixel_values"].size(0)
