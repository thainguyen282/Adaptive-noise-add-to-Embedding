"""CLI pipeline for high-entropy token discovery on DAPO prompts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from dapo_token_importance.data import DEFAULT_DATASET, DapoSample, iter_dapo_samples, load_dapo_dataset
from dapo_token_importance.scoring import TokenScore, build_token_scores, entropy_from_logits


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-entropy tokens in DAPO prompts.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Causal LM checkpoint for entropy scoring.")
    parser.add_argument("--output", required=True, help="Path to write JSONL results.")
    parser.add_argument("--max-samples", type=int, default=32, help="Maximum number of samples to score.")
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum tokenized prompt length.")
    parser.add_argument("--importance-ratio", type=float, default=0.2, help="Top entropy ratio per sample.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Keep tokenizer special tokens in the importance ranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    dataset = load_dapo_dataset(args.dataset, split=args.split, max_samples=args.max_samples)
    samples = iter_dapo_samples(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in tqdm(list(batched(samples, args.batch_size)), desc="Scoring prompts"):
            for row in score_batch(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                importance_ratio=args.importance_ratio,
                include_special_tokens=args.include_special_tokens,
            ):
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if not 0 < args.importance_ratio <= 1:
        raise ValueError("--importance-ratio must be in (0, 1]")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def batched(samples: list[DapoSample], batch_size: int) -> Iterator[list[DapoSample]]:
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


@torch.inference_mode()
def score_batch(
    batch: list[DapoSample],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
    importance_ratio: float,
    include_special_tokens: bool,
) -> list[dict[str, Any]]:
    prompt_texts = [render_prompt_for_model(sample, tokenizer) for sample in batch]
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model(**encoded, use_cache=False)
    shifted_entropy = entropy_from_logits(outputs.logits[:, :-1, :]).cpu()
    target_ids = encoded["input_ids"][:, 1:].cpu()
    target_mask = encoded["attention_mask"][:, 1:].bool().cpu()

    results: list[dict[str, Any]] = []
    special_ids = set(tokenizer.all_special_ids)

    for row_idx, sample in enumerate(batch):
        valid_positions = target_mask[row_idx].nonzero(as_tuple=False).squeeze(1)
        token_ids: list[int] = []
        token_indices: list[int] = []
        entropy_values: list[torch.Tensor] = []

        for position in valid_positions.tolist():
            token_id = int(target_ids[row_idx, position].item())
            original_index = position + 1
            if not include_special_tokens and token_id in special_ids:
                continue
            token_ids.append(token_id)
            token_indices.append(original_index)
            entropy_values.append(shifted_entropy[row_idx, position])

        if not entropy_values:
            token_scores: list[TokenScore] = []
        else:
            token_entropy = torch.stack(entropy_values)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            token_scores = build_token_scores(
                token_ids=token_ids,
                tokens=tokens,
                entropy=token_entropy,
                top_ratio=importance_ratio,
                token_indices=token_indices,
            )

        results.append(format_result(sample, token_scores, prompt_texts[row_idx]))

    return results


def render_prompt_for_model(sample: DapoSample, tokenizer: PreTrainedTokenizerBase) -> str:
    """Use the tokenizer chat template when possible, otherwise fall back to plain text."""

    if sample.prompt_messages is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                sample.prompt_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return sample.prompt_text
    return sample.prompt_text


def format_result(sample: DapoSample, token_scores: list[TokenScore], model_prompt: str) -> dict[str, Any]:
    selected_scores = [score for score in token_scores if score.selected]

    return {
        "sample_id": sample.sample_id,
        "data_source": sample.data_source,
        "ground_truth": sample.ground_truth,
        "prompt_text": sample.prompt_text,
        "model_prompt": model_prompt,
        "token_indices": [score.index for score in token_scores],
        "token_ids": [score.token_id for score in token_scores],
        "tokens": [score.token for score in token_scores],
        "entropy_scores": [score.entropy for score in token_scores],
        "selected_token_indices": [score.index for score in selected_scores],
        "selected_token_ids": [score.token_id for score in selected_scores],
        "selected_tokens": [score.token for score in selected_scores],
        "selected_entropy_scores": [score.entropy for score in selected_scores],
    }


if __name__ == "__main__":
    main()
