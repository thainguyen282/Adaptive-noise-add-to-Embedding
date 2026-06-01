"""Dataset loading and prompt normalization for VERL-style DAPO data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset


DEFAULT_DATASET = "sungyub/dapo-math-17k-verl"


@dataclass(frozen=True)
class DapoSample:
    """A normalized DAPO sample ready for token scoring."""

    sample_id: int
    prompt_text: str
    prompt_messages: list[dict[str, str]] | None
    data_source: str | None
    ground_truth: str | None


def load_dapo_dataset(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Load a Hugging Face DAPO dataset split."""

    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive when provided")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def normalize_dapo_sample(record: dict[str, Any], sample_id: int) -> DapoSample:
    """Normalize one VERL-style record into prompt text and metadata."""

    prompt = record.get("prompt")
    prompt_messages = _normalize_prompt_messages(prompt)
    prompt_text = _prompt_to_text(prompt, prompt_messages)
    if not prompt_text.strip():
        raise ValueError(f"Sample {sample_id} has an empty prompt")

    reward_model = record.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None

    return DapoSample(
        sample_id=sample_id,
        prompt_text=prompt_text,
        prompt_messages=prompt_messages,
        data_source=record.get("data_source"),
        ground_truth=ground_truth,
    )


def iter_dapo_samples(dataset: Dataset) -> list[DapoSample]:
    """Convert a loaded dataset into normalized samples."""

    return [normalize_dapo_sample(record, idx) for idx, record in enumerate(dataset)]


def _normalize_prompt_messages(prompt: Any) -> list[dict[str, str]] | None:
    if not isinstance(prompt, list):
        return None

    messages: list[dict[str, str]] = []
    for message in prompt:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return None
        messages.append({"role": str(role), "content": str(content)})
    return messages


def _prompt_to_text(prompt: Any, prompt_messages: list[dict[str, str]] | None) -> str:
    if prompt_messages is not None:
        return "\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)
    if isinstance(prompt, str):
        return prompt
    return str(prompt)
