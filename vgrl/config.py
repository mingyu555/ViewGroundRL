"""Configuration for the view-grounding term."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ViewGroundingConfig:
    enabled: bool = True

    mode: str = "kl_term"
    """"kl_term"       기존 방식. L_view = -coef*(kl1-kl2). 3B 깨끗한 split 에서
                       F1 -0.0077, gap 회복 +0.0048(0.55 sigma) - 기여 구간 없음.
                       구조적 결함: '근거뷰 없을 때 못하는 것'을 보상해서 모델이
                       취약해지는 방식으로도 목적을 채운다.
        "rollout_mask" 대조뷰 불변성. GRPO 그룹의 일부 롤아웃을 '대조뷰 셀을 가린'
                       입력에서 뽑고 보상 함수는 그대로 둔다. 가린 조건에서 맞힌 답은
                       양의 어드밴티지, 틀린 답은 음의 어드밴티지를 받아
                       '무관한 뷰가 빠져도 맞혀라'가 직접 학습된다.
                       추가 생성이 없어 비용은 plain GRPO 와 같다."""

    masked_rollouts: int = 2
    """rollout_mask 모드에서 num_generations 중 몇 개를 가린 입력으로 뽑을지."""

    normalize_per_condition: bool = True
    """조건별 어드밴티지 정규화. 두 조건을 한 그룹으로 묶으면 가린 롤아웃이 평균을
    낮춰 전체뷰 롤아웃의 어드밴티지가 부풀려진다. num_generations 8 중 2 를 가리는
    경우 편향이 작지 않다."""

    grid_cells: int = 6
    """합성 격자의 셀 수. DriveLMM-o1 은 2x3 격자 1장을 입력으로 쓰므로
    '뷰를 가린다'가 '격자 셀을 가린다'가 된다 (6장 따로가 아니다)."""

    grid_cols: int = 3

    grid_input: bool = False
    """입력이 '6뷰를 합친 격자 1장'인가. True 면 마스킹이 이미지 단위가 아니라
    셀 단위로 동작한다 (DriveLMM-o1 이 이 형태다)."""

    coef: float = 0.02
    """Weight on the margin term.

    PAPO (arXiv:2507.06448) sets its analogous gamma to 0.02 for PAPO-GRPO at 3B
    and 7B, 0.01 for PAPO-DAPO, and ablates 0.005-0.04, reporting severe model
    collapse at 0.04. Our base and algorithm match PAPO-GRPO-3B, so 0.02 is the
    like-for-like value; 0.005 is the bottom of their ablation range.

    Two differences argue for a *larger* coef here, not smaller: PAPO corrupts
    60-80% of patches at random, while we blank one whole view (~17% of the image
    budget), so the same gamma buys a smaller divergence; and our term is a
    difference of two KLs, which is smaller than either.

    PAPO also pairs gamma with a Double Entropy Loss (eta 0.03-0.05) against the
    loss-hacking mode. We have no such regulariser — `kl_clip` stands in for it —
    which is the likelier explanation for the runaway we saw on DriveLM at 0.02
    than gamma being too high."""

    sign: str = "maximize_margin"
    """"maximize_margin"  loss = -coef * (kl1 - kl2)   <- trains grounding
        "literal"         loss = +coef * (kl1 - kl2)   <- ablation only; this
                          rewards ignoring the evidence view
        "kl1_only"        loss = -coef * kl1           <- PAPO, no control branch"""

    num_views: int = 6
    num_views_to_mask: int = 1
    """How many views each branch blanks. Both branches always blank the same
    count, so the contrast isolates *which* view was removed."""

    mask_mode: str = "black"
    """"black" (normalised all-zero pixels), "zero" (zeros in normalised space),
    or "noise"."""
    noise_std: float = 1.0

    answer_view_source: str = "both"
    """Which text the evidence views are read from: "answer", "question", "both"."""

    estimator: str = "k3"
    """"k3"  r - log r - 1   — 항상 >= 0. 끝난 실행들이 쓴 값이라 기본값으로 남긴다.
       "k1"  -log r          — 방향성. 진단이 쓰는 값.
    NuInstruct 실측에서 둘이 반대 결론을 낸다: 학습 전 모델이 k3 로는 kl1/kl2 =
    0.839/0.255 (3.3배 선택성) 인데 k1 로는 -0.030/-0.029 (선택성 없음) 이다.
    k3 는 "분포가 흔들렸다" 를 재고 k1 은 "정답이 어려워졌다" 를 재며, 문제 정의가
    말하는 것은 후자다."""

    token_filter: str = "all"
    """관점항을 걸 완성 토큰 범위. "all" 전부 / "content" 내용어만.
    실측: 내용어가 NuInstruct 62%, OmniDrive 45%. 전부에 걸면 신호가 1.5배 희석되고,
    softplus 팔에서는 비내용어 낙폭이 내용어의 1.2배까지 따라붙었다(SFT 는 9.5배 차이).
    `the` 가 뷰에 의존한다는 건 접지가 아니라 전역 확신 이동이므로 손실에서 뺀다."""

    transform: str = "linear"
    """"linear"   L = -coef * (kl1 - kl2)          — 지금까지의 형태
       "softplus" L = coef * softplus(kl2 - kl1)  — InfoNCE 형태
    k1 추정기에서 margin 은 2지선다 로짓 차이(log p(대조가림) - log p(근거가림))라,
    softplus(-margin) 은 "올바른 분기를 고를 확률"의 교차엔트로피가 된다. 기울기가
    -sigmoid(-margin) 으로 1 이하이고 margin 이 커지면 사라져서, 선형 형태가 냈던
    폭주(NuInstruct 실측 k3 kl1 3.8e13, 방향성 낙폭 +4.64 nats)가 구조적으로 막힌다.
    margin 0 근처에서는 선형에 계수 1/2 이므로 coef 를 대략 두 배로 잡아야 압력이 맞다."""

    margin_clip: float | None = 10.0
    kl_clip: float | None = None
    """Guards against the KL-hacking mode PAPO documents, where the model inflates
    the divergence instead of grounding on the view."""

    detach_control_branch: bool = False
    """If True, stop gradients through the control (kl2) pass, making it a pure
    baseline. Cheaper backward, but no longer penalises rising kl2 directly."""

    warmup_steps: int = 0
    """Delay the term so the policy first stabilises under plain GRPO."""

    seed: int = 0

    log_prefix: str = "vg"

    def validate(self) -> None:
        if self.mode not in {"kl_term", "rollout_mask"}:
            raise ValueError(f"bad mode: {self.mode!r}")
        if self.mode == "rollout_mask" and self.masked_rollouts < 1:
            raise ValueError("rollout_mask 모드는 masked_rollouts >= 1 이어야 한다")
        if self.token_filter not in {"all", "content"}:
            raise ValueError(f"bad token_filter: {self.token_filter!r}")
        if self.transform not in {"linear", "softplus"}:
            raise ValueError(f"bad transform: {self.transform!r}")
        if self.estimator not in {"k3", "k1"}:
            raise ValueError(f"bad estimator: {self.estimator!r}")
        if self.sign not in {"maximize_margin", "literal", "kl1_only"}:
            raise ValueError(f"bad sign: {self.sign!r}")
        if self.mask_mode not in {"black", "zero", "noise"}:
            raise ValueError(f"bad mask_mode: {self.mask_mode!r}")
        if self.answer_view_source not in {"answer", "question", "both"}:
            raise ValueError(f"bad answer_view_source: {self.answer_view_source!r}")
        if self.num_views_to_mask < 1:
            raise ValueError("num_views_to_mask must be >= 1")
        if self.num_views_to_mask * 2 > self.num_views:
            raise ValueError(
                f"num_views_to_mask={self.num_views_to_mask} leaves no room for a "
                f"matched control set out of {self.num_views} views"
            )
