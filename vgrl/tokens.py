"""내용어 토큰 판별. 진단(tools/measure_view_kl.py)과 학습이 같은 정의를 쓰게 한다.

측정으로 확인한 것: 관점항을 완성 토큰 전부에 걸면 관사·전치사·구두점이 평균을
희석한다 (NuInstruct 정답 33토큰 중 내용어 20.6 = 62%). 그리고 softplus 팔에서는
비내용어 낙폭이 +1.976, 내용어가 +2.386 으로 1.2배 차이밖에 안 났다 — SFT 는
9.5배, 일반 GRPO 는 12.5배였다. `the` 가 내용어만큼 뷰에 의존한다는 것은 접지가
아니라 전역적 확신 이동이므로, 손실이 그 토큰들을 밀지 않도록 잘라낸다.
"""

from __future__ import annotations

import torch

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "and", "or", "but", "that", "this", "these", "those", "it", "its",
    "there", "here", "as", "if", "than", "then", "so", "which", "you",
    "your", "we", "our", "us", "will", "would", "should", "could", "can",
    "may", "might", "must", "have", "has", "had", "do", "does", "did",
    "not", "no", "also", "while", "however", "additionally", "firstly",
    "scenario", "following", "given", "current", "vehicle", "ego",
}
PUNCT = set(".,;:!?()[]{}<>-—–\"'`/\\|+*=&%$#@~^_")

_CACHE: dict[int, torch.Tensor] = {}


def _is_content(text: str) -> bool:
    w = text.strip().lower()
    if not w:
        return False
    if w in STOP:
        return False
    return not all(c in PUNCT or c.isspace() for c in w)


def content_vocab_mask(tokenizer) -> torch.Tensor:
    """어휘 전체에 대한 bool 표. 스텝마다 토큰을 디코드하면 느리므로 한 번만 만든다."""
    key = id(tokenizer)
    if key not in _CACHE:
        n = len(tokenizer)
        flags = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            try:
                flags[i] = _is_content(tokenizer.decode([i]))
            except Exception:
                flags[i] = False
        _CACHE[key] = flags
    return _CACHE[key]


def content_token_mask(ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """ids 와 같은 모양의 bool 마스크. 내용어 위치만 True."""
    table = content_vocab_mask(tokenizer).to(ids.device)
    safe = ids.clamp(min=0, max=table.shape[0] - 1)
    return table[safe]
