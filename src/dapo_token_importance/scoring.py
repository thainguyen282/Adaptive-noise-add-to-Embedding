"""Token entropy scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenScore:
    """Entropy score for one token in one prompt."""

    index: int
    token_id: int
    token: str
    entropy: float
    selected: bool


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy from logits along the vocabulary dimension."""

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def top_entropy_mask(entropy: torch.Tensor, top_ratio: float) -> torch.Tensor:
    """Return a boolean mask selecting the highest-entropy values."""

    if not 0 < top_ratio <= 1:
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
    if entropy.numel() == 0:
        return torch.zeros_like(entropy, dtype=torch.bool)

    top_k = max(1, int(entropy.numel() * top_ratio + 0.9999))
    _, top_indices = torch.topk(entropy, k=top_k)
    mask = torch.zeros_like(entropy, dtype=torch.bool)
    mask[top_indices] = True
    return mask


def build_token_scores(
    token_ids: list[int],
    tokens: list[str],
    entropy: torch.Tensor,
    top_ratio: float,
    token_indices: list[int] | None = None,
) -> list[TokenScore]:
    """Combine token ids, decoded strings, and entropy into serializable scores."""

    if len(token_ids) != len(tokens) or len(tokens) != entropy.numel():
        raise ValueError("token_ids, tokens, and entropy must have the same length")
    if token_indices is None:
        token_indices = list(range(len(tokens)))
    if len(token_indices) != len(tokens):
        raise ValueError("token_indices must have the same length as tokens")

    selected_mask = top_entropy_mask(entropy, top_ratio).tolist()
    entropy_values = entropy.detach().cpu().tolist()

    return [
        TokenScore(
            index=token_indices[index],
            token_id=token_id,
            token=token,
            entropy=float(entropy_values[index]),
            selected=bool(selected_mask[index]),
        )
        for index, (token_id, token) in enumerate(zip(token_ids, tokens, strict=True))
    ]
