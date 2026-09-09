"""Cross-View Geometric Enabler (CVGE) — VGGDrive 의 stage 1 모듈.

VGGDrive (CVPR 2026, arXiv:2602.20794) 4.1:
  "The training process consists of two stages, with the parameters of the VGGT
   frozen throughout both stages. In the first stage, we freeze the base VLM
   parameters and train only the parameters introduced by CVGE for 2 epochs,
   using a learning rate of 1e-4 and a batch size of 2."

원본 구현(inject_utils/Qwen2_5_vggt_fusion_inject.py)은 Qwen2_5_VLModel.forward 를
통째로 복사해 transformers 4.49 의 디코더 루프 안에 주입을 끼워 넣는다. 이 레포는
transformers 5.13 이고 그 사이에 텍스트 스택이 Qwen2_5_VLTextModel 로 분리돼서
그 파일은 그대로 쓸 수 없다. 여기서는 같은 연산을 디코더 레이어 forward hook 으로
붙인다 — 수식은 동일하고 모델 내부 구현에 묶이지 않는다.

레이어 i 마다, 이미지 토큰 위치에서만:
    h   = hidden[img_mask]                       [B, N, 3584]
    q   = down_llm(h)                            [B, N, 512]
    kv  = down_3d(f3d)                           [B, S*T, 512]
    z   = CrossAttn(q, kv)                       [B, N, 512]
    hidden[img_mask] = h + up(z)                 [B, N, 3584]

원본과 다른 점은 두 가지뿐이다:
  - softmax/matmul 을 직접 쓰지 않고 scaled_dot_product_attention 을 쓴다. 수식은
    같지만 [B,h,N,S*T] 점수 행렬을 실체화하지 않는다. 여기서 N=2752(2x3 격자),
    S*T=4662(6뷰) 라 레이어당 점수만 bf16 으로 97MB 고, 28 레이어를 역전파용으로
    들고 있으면 2.7GB 가 그냥 날아간다.
  - 마스크 인덱싱을 배치 루프 대신 벡터화했다. 원본은 파이썬 for 문으로 배치를 돈다.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- VGGT backbone

_VGGT_SINGLETON = {}


def _ensure_vggt_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tp = os.path.join(root, "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)


def load_vggt(weights: str, device, dtype=torch.bfloat16):
    """VGGT-1B 를 얼린 채로 한 번만 올린다.

    프로세스당 하나만 쓴다. 6 뷰 aggregator 만 돌리므로 head 는 필요 없지만,
    체크포인트가 통짜라 전체를 올린 뒤 aggregator 만 호출한다.
    """
    key = (weights, str(device), str(dtype))
    if key in _VGGT_SINGLETON:
        return _VGGT_SINGLETON[key]

    _ensure_vggt_on_path()
    from vggt.models.vggt import VGGT

    model = VGGT()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("aggregator") for k in sd):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    agg_missing = [k for k in missing if k.startswith("aggregator")]
    if agg_missing:
        raise RuntimeError(
            f"VGGT aggregator 가중치가 {len(agg_missing)}개 비어 있다: {agg_missing[:5]}"
        )
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    _VGGT_SINGLETON[key] = model
    return model


@torch.no_grad()
def vggt_features(vggt, images: torch.Tensor) -> torch.Tensor:
    """[B, S, 3, H, W] -> [B, S*T, 2048] 마지막 aggregator 토큰.

    VGGT aggregator 는 프레임내 어텐션과 전역 어텐션 출력을 채널로 이어 붙여
    1024*2 = 2048 을 낸다. VGGDrive 의 in_dim=2048 이 이 값이다.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens_list, _ = vggt.aggregator(images)
    feat = tokens_list[-1]                       # [B, S, T, 2048]
    return feat.flatten(1, 2).detach()           # [B, S*T, 2048]


# ------------------------------------------------------------------ CVGE block


class CrossAttentionFusion(nn.Module):
    """VGGDrive CrossAttentionFusion. Q=VLM 은닉, K/V=VGGT 기하 토큰."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, f_llm: torch.Tensor, f_3d: torch.Tensor) -> torch.Tensor:
        b, n, _ = f_llm.shape
        m = f_3d.shape[1]

        q = self.q_proj(f_llm).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(f_3d).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(f_3d).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.layernorm(f_llm + self.out_proj(out))


class CVGEBlock(nn.Module):
    """레이어 하나에 붙는 CVGE. 원본 prompt_tuning_mlp[i] 의 4개 원소와 같다."""

    def __init__(self, hidden_size: int = 3584, in_dim: int = 2048,
                 scale: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        mid = in_dim // scale                              # 512
        self.down_llm = nn.Sequential(
            nn.Linear(hidden_size, mid), nn.GELU(), nn.Linear(mid, mid),
        )
        self.down_3d = nn.Sequential(
            nn.Linear(in_dim, mid), nn.GELU(), nn.Linear(mid, mid),
        )
        self.fuse = CrossAttentionFusion(dim=mid, num_heads=num_heads, dropout=dropout)
        self.up = nn.Sequential(
            nn.Linear(mid, mid), nn.GELU(), nn.Linear(mid, hidden_size),
        )
        # 출력 투영을 0 으로 초기화한다. 그러면 CVGE 가 처음에 정확히 항등원이라
        # 학습이 얼어붙은 VLM 의 손실에서 그대로 출발한다. 원본 VGGDrive 구현은
        # 이걸 안 하는데, 그 상태로 4 GPU 학습을 돌리면 19 스텝에서 터진다:
        # 28 개 레이어 전부에 무작위 잔차가 들어가 초기 grad_norm 이 56.9 까지
        # 뜨고, 그래디언트 하나가 inf 가 되는 순간 clip_grad_norm_ 의 계수가
        # 1.0/inf = 0 이 되면서 inf * 0 = nan 으로 전 파라미터가 오염된다.
        # LoRA 의 B 행렬, ControlNet 의 zero-conv 와 같은 처방이다.
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, h_img: torch.Tensor, f_3d: torch.Tensor) -> torch.Tensor:
        z = self.fuse(self.down_llm(h_img), self.down_3d(f_3d))
        return h_img + self.up(z)                          # residual


class CVGE(nn.Module):
    """28 레이어분 CVGE + 디코더 hook 설치.

    학습 대상은 이 모듈뿐이다. VLM 도 VGGT 도 건드리지 않는다.
    """

    def __init__(self, hidden_size: int, num_layers: int, in_dim: int = 2048,
                 scale: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CVGEBlock(hidden_size, in_dim, scale, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # forward 마다 훅이 읽어 갈 자리. 배치 단위로 세팅한다.
        self._f3d: torch.Tensor | None = None
        self._img_mask: torch.Tensor | None = None
        self._handles: list = []

    # -- 훅이 읽는 배치 상태 ------------------------------------------------
    def set_batch(self, f_3d: torch.Tensor | None, img_mask: torch.Tensor | None) -> None:
        self._f3d = f_3d
        self._img_mask = img_mask

    def clear_batch(self) -> None:
        self._f3d = None
        self._img_mask = None

    # -- 훅 설치 -------------------------------------------------------------
    def attach(self, decoder_layers) -> None:
        """decoder_layers 각 원소의 출력에 CVGE 를 끼운다."""
        assert len(decoder_layers) == len(self.blocks), (
            f"레이어 수 불일치: 모델 {len(decoder_layers)} vs CVGE {len(self.blocks)}"
        )
        self.detach()
        for idx, layer in enumerate(decoder_layers):
            self._handles.append(
                layer.register_forward_hook(self._make_hook(idx))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            if self._f3d is None or self._img_mask is None:
                return output
            # transformers 5.x 의 디코더 레이어는 Tensor 또는 튜플을 낸다.
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output

            mask = self._img_mask                      # [B, L] bool
            # 디코딩 단계에서는 hidden 이 [B, 1, H] 로 들어와 마스크와 길이가
            # 어긋난다. 이미지 토큰은 프리필에만 있으므로 그때만 주입한다.
            if mask.shape[1] != hidden.shape[1]:
                return output
            # 샘플마다 이미지 토큰 수가 같아야 [B, N, H] 로 모을 수 있다.
            # 한 샘플에 격자 이미지 하나뿐이라 실제로 항상 같다.
            n = int(mask[0].sum())
            if n == 0:
                return output
            b, _, hsz = hidden.shape
            h_img = hidden[mask].view(b, n, hsz)

            f3d = self._f3d.to(dtype=h_img.dtype, device=h_img.device)
            new_img = self.blocks[idx](h_img, f3d).to(hidden.dtype)

            hidden = hidden.masked_scatter(mask.unsqueeze(-1), new_img)
            return (hidden,) + output[1:] if is_tuple else hidden

        return hook
