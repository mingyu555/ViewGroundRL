"""Mask out whole camera views inside an already-processed `pixel_values` batch.

Qwen2.5-VL flattens every image into patch rows and concatenates them, so a batch
carries `pixel_values` of shape [total_patch_rows, patch_dim] plus
`image_grid_thw` [total_images, 3] and a per-sample `num_images`. The number of
rows contributed by image *j* is `image_grid_thw[j].prod()`, which makes the row
span of a given view exactly addressable — we can blank one camera without
re-running the image processor.

Masking is done in *normalised* patch space. The row that a fully black image
produces is derived once from the processor itself rather than assumed, so this
stays correct if the patch memory layout or the normalisation constants change.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _black_patch_row(processor, patch_dim: int, device, dtype) -> torch.Tensor:
    """The single patch row that a fully black image yields under `processor`.

    Every patch of an all-black image is identical, so row 0 of the processed
    output is the value we need. Falls back to an analytic construction (all
    channels at `(0 - mean) / std`) if the processor cannot be called.
    """
    from PIL import Image

    image_processor = getattr(processor, "image_processor", processor)
    try:
        patch = getattr(image_processor, "patch_size", 14)
        black = Image.new("RGB", (patch * 4, patch * 4), (0, 0, 0))
        out = image_processor(images=black, return_tensors="pt")
        row = out["pixel_values"][0]
        if row.numel() == patch_dim:
            return row.to(device=device, dtype=dtype)
    except Exception:
        pass

    mean = torch.tensor(getattr(image_processor, "image_mean", [0.0, 0.0, 0.0]))
    std = torch.tensor(getattr(image_processor, "image_std", [1.0, 1.0, 1.0]))
    per_channel = (-mean / std).to(device=device, dtype=dtype)  # value of black
    if patch_dim % per_channel.numel() != 0:
        raise ValueError(f"patch_dim {patch_dim} not divisible by {per_channel.numel()} channels")
    # channel is the outermost axis of a flattened Qwen2-VL patch row
    return per_channel.repeat_interleave(patch_dim // per_channel.numel())


class ViewMasker:
    """Builds masked copies of `pixel_values` with selected views blanked."""

    def __init__(self, processor, mode: str = "black", noise_std: float = 1.0):
        if mode not in {"black", "zero", "noise"}:
            raise ValueError(f"unknown mask mode: {mode!r}")
        self.processor = processor
        self.mode = mode
        self.noise_std = noise_std
        self._row_cache: dict[tuple, torch.Tensor] = {}

    def _fill_row(self, patch_dim: int, device, dtype) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(patch_dim, device=device, dtype=dtype)
        if self.mode == "noise":
            return torch.randn(patch_dim, device=device, dtype=dtype) * self.noise_std
        key = (patch_dim, str(device), str(dtype))
        if key not in self._row_cache:
            self._row_cache[key] = _black_patch_row(self.processor, patch_dim, device, dtype)
        return self._row_cache[key]

    @staticmethod
    def row_spans(
        image_grid_thw: torch.Tensor,
        num_images: Sequence[int],
    ) -> list[list[tuple[int, int]]]:
        """`spans[i][v]` = (row_start, row_end) of view v of sample i."""
        rows_per_image = image_grid_thw.prod(dim=-1).tolist()
        cum_rows = [0]
        for r in rows_per_image:
            cum_rows.append(cum_rows[-1] + int(r))

        spans: list[list[tuple[int, int]]] = []
        img_offset = 0
        for n in num_images:
            spans.append(
                [(cum_rows[img_offset + v], cum_rows[img_offset + v + 1]) for v in range(n)]
            )
            img_offset += n
        return spans

    def mask_grid_cells(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_images: Sequence[int],
        views_per_sample: Sequence[Sequence[int]],
        grid_rows: int = 2,
        grid_cols: int = 3,
    ) -> torch.Tensor:
        """뷰가 '이미지'가 아니라 '한 장 안의 격자 셀'일 때의 마스킹.

        DriveLMM-o1 은 6뷰를 2x3 격자 한 장으로 합쳐 넣으므로 이미지 단위 span 을
        지울 수 없다. Qwen2.5-VL 은 image_grid_thw=(t,h,w) 로 패치 격자를 주고
        패치는 행 우선으로 펼쳐지므로, 셀 (r,c) 는
            행 [r*h/rows, (r+1)*h/rows) x 열 [c*w/cols, (c+1)*w/cols)
        에 해당한다. 그 좌표를 1차원 인덱스로 바꿔 지운다.
        """
        out = pixel_values.clone()
        spans = self.row_spans(image_grid_thw, num_images)
        dim = pixel_values.size(-1)
        fill = self._fill_row(dim, pixel_values.device, pixel_values.dtype)
        for i, cells in enumerate(views_per_sample):
            if not cells or not spans[i]:
                continue
            start, end = spans[i][0]          # 합성 격자는 이미지 1장
            t, h, w = (int(x) for x in image_grid_thw[sum(num_images[:i])])
            if (end - start) != t * h * w:
                raise RuntimeError(
                    f"sample {i}: span {end - start} != t*h*w {t * h * w}; "
                    "패치 배치 가정이 깨졌다"
                )
            rh, cw = h // grid_rows, w // grid_cols
            for cell in cells:
                r, c = divmod(int(cell), grid_cols)
                if r >= grid_rows:
                    raise IndexError(f"cell {cell} out of {grid_rows}x{grid_cols}")
                for frame in range(t):
                    base = start + frame * h * w
                    for row in range(r * rh, min((r + 1) * rh, h)):
                        lo = base + row * w + c * cw
                        hi = base + row * w + min((c + 1) * cw, w)
                        out[lo:hi] = fill
        return out

    def mask(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_images: Sequence[int],
        views_per_sample: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Return a copy of `pixel_values` with the requested views blanked.

        `views_per_sample[i]` lists view indices to blank for sample i; an empty
        list leaves that sample untouched.
        """
        if len(views_per_sample) != len(num_images):
            raise ValueError(
                f"views_per_sample has {len(views_per_sample)} entries but there are "
                f"{len(num_images)} samples"
            )
        out = pixel_values.clone()
        spans = self.row_spans(image_grid_thw, num_images)
        fill = self._fill_row(pixel_values.size(-1), pixel_values.device, pixel_values.dtype)

        for i, views in enumerate(views_per_sample):
            for v in views:
                if v < 0 or v >= len(spans[i]):
                    raise IndexError(
                        f"sample {i} has {len(spans[i])} images; view index {v} out of range"
                    )
                start, end = spans[i][v]
                out[start:end] = (
                    self._fill_row(pixel_values.size(-1), pixel_values.device, pixel_values.dtype)
                    if self.mode == "noise"
                    else fill
                )
        return out
