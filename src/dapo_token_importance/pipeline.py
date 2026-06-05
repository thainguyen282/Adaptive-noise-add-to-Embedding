"""CLI pipeline for high-entropy token discovery on DAPO prompts."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from dapo_token_importance.data import DEFAULT_DATASET, DapoSample, iter_dapo_samples, load_dapo_dataset
from dapo_token_importance.scoring import TokenScore, build_token_scores, entropy_from_logits, top_entropy_mask


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


@dataclass
class ScoredPrompt:
    """A prompt with token scores before final export."""

    sample: DapoSample
    model_prompt: str
    generated_response: str
    token_scores: list[TokenScore]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-entropy tokens in DAPO prompts.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--entropy-model",
        default=DEFAULT_MODEL,
        help="Model checkpoint for local entropy scoring. Also used as the vLLM generation model by default.",
    )
    parser.add_argument("--model", default=None, help="Optional generation model override for vLLM/local generation.")
    parser.add_argument("--output", required=True, help="Path to write JSONL results.")
    parser.add_argument(
        "--selected-output",
        default=None,
        help="Path to write a compact JSON list of all selected tokens. Defaults to <output>.selected.json.",
    )
    parser.add_argument("--max-samples", type=int, default=32, help="Maximum number of samples to score.")
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size.")
    parser.add_argument(
        "--score-micro-batch-size",
        type=int,
        default=1,
        help="Micro-batch size for the logits/entropy forward pass. Keep small to avoid OOM.",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum tokenized prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum generated response length.")
    parser.add_argument(
        "--entropy-backend",
        choices=["local", "vllm"],
        default="local",
        help="Compute entropy locally with Transformers or approximately from vLLM top logprobs.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=["local", "vllm"],
        default="local",
        help="Use local Transformers generation or a vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API.",
    )
    parser.add_argument(
        "--vllm-timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for each vLLM generation request.",
    )
    parser.add_argument(
        "--vllm-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs vLLM should return per generated token for approximate entropy.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used to generate reasoning responses.",
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability.")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to sample generated responses. Use --no-do-sample for greedy decoding.",
    )
    parser.add_argument(
        "--importance-ratio",
        type=float,
        default=0.2,
        help="Top entropy ratio across all valid tokens in the full scored dataset.",
    )
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

    dataset = load_dapo_dataset(args.dataset, split=args.split, max_samples=args.max_samples)
    samples = iter_dapo_samples(dataset)

    device = resolve_device(args.device)
    entropy_model_name = args.entropy_model
    generation_model_name = args.model or args.entropy_model
    tokenizer = AutoTokenizer.from_pretrained(entropy_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model: PreTrainedModel | None = None
    if args.entropy_backend == "local":
        model = AutoModelForCausalLM.from_pretrained(
            entropy_model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

    scored_prompts: list[ScoredPrompt] = []
    for batch in tqdm(list(batched(samples, args.batch_size)), desc="Scoring prompts"):
        scored_prompts.extend(
            score_batch(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                score_micro_batch_size=args.score_micro_batch_size,
                max_new_tokens=args.max_new_tokens,
                generation_model_name=generation_model_name,
                generation_backend=args.generation_backend,
                entropy_backend=args.entropy_backend,
                vllm_base_url=args.vllm_base_url,
                vllm_timeout=args.vllm_timeout,
                vllm_logprobs=args.vllm_logprobs,
                generation_temperature=args.generation_temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                include_special_tokens=args.include_special_tokens,
            )
        )
        release_cuda_memory(device)

    apply_dataset_global_selection(scored_prompts, args.importance_ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for scored_prompt in scored_prompts:
            row = format_result(
                scored_prompt.sample,
                scored_prompt.token_scores,
                scored_prompt.model_prompt,
                scored_prompt.generated_response,
            )
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_output_path = resolve_selected_output_path(output_path, args.selected_output)
    selected_output_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_output_path.open("w", encoding="utf-8") as selected_output_file:
        json.dump(
            format_selected_tokens(scored_prompts),
            selected_output_file,
            ensure_ascii=False,
            indent=2,
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.score_micro_batch_size < 1:
        raise ValueError("--score-micro-batch-size must be positive")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.vllm_timeout <= 0:
        raise ValueError("--vllm-timeout must be positive")
    if args.vllm_logprobs < 1:
        raise ValueError("--vllm-logprobs must be positive")
    if args.entropy_backend == "vllm" and args.generation_backend != "vllm":
        raise ValueError("--entropy-backend vllm requires --generation-backend vllm")
    if args.generation_temperature <= 0:
        raise ValueError("--generation-temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
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


def resolve_selected_output_path(output_path: Path, selected_output: str | None) -> Path:
    if selected_output is not None:
        return Path(selected_output)
    if output_path.suffix:
        return output_path.with_suffix(f"{output_path.suffix}.selected.json")
    return output_path.with_name(f"{output_path.name}.selected.json")


@torch.inference_mode()
def score_batch(
    batch: list[DapoSample],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel | None,
    device: torch.device,
    max_length: int,
    score_micro_batch_size: int,
    max_new_tokens: int,
    generation_model_name: str,
    generation_backend: str,
    entropy_backend: str,
    vllm_base_url: str,
    vllm_timeout: float,
    vllm_logprobs: int,
    generation_temperature: float,
    top_p: float,
    do_sample: bool,
    include_special_tokens: bool,
) -> list[ScoredPrompt]:
    prompt_texts = [render_prompt_for_model(sample, tokenizer, add_generation_prompt=True) for sample in batch]

    if entropy_backend == "vllm":
        return generate_and_score_with_vllm_logprobs(
            batch=batch,
            prompt_texts=prompt_texts,
            model_name=generation_model_name,
            base_url=vllm_base_url,
            timeout=vllm_timeout,
            max_new_tokens=max_new_tokens,
            temperature=generation_temperature,
            top_p=top_p,
            do_sample=do_sample,
            logprobs=vllm_logprobs,
        )

    if model is None:
        raise ValueError("Local entropy scoring requires a loaded Transformers model")

    if generation_backend == "vllm":
        generated_responses = generate_with_vllm(
            prompt_texts=prompt_texts,
            model_name=generation_model_name,
            base_url=vllm_base_url,
            timeout=vllm_timeout,
            max_new_tokens=max_new_tokens,
            temperature=generation_temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        return score_prompt_response_batch(
            batch=batch,
            prompt_texts=prompt_texts,
            generated_responses=generated_responses,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            score_micro_batch_size=score_micro_batch_size,
            include_special_tokens=include_special_tokens,
        )

    return generate_and_score_local_batch(
        batch=batch,
        prompt_texts=prompt_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        generation_temperature=generation_temperature,
        top_p=top_p,
        do_sample=do_sample,
        score_micro_batch_size=score_micro_batch_size,
        include_special_tokens=include_special_tokens,
    )


def generate_and_score_local_batch(
    batch: list[DapoSample],
    prompt_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    generation_temperature: float,
    top_p: float,
    do_sample: bool,
    score_micro_batch_size: int,
    include_special_tokens: bool,
) -> list[ScoredPrompt]:
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    generation_kwargs: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = generation_temperature
        generation_kwargs["top_p"] = top_p

    generated_ids = model.generate(**generation_kwargs)
    release_cuda_memory(device)
    generated_attention_mask = build_generated_attention_mask(
        generated_ids=generated_ids,
        prompt_attention_mask=encoded["attention_mask"],
        prompt_width=prompt_width,
        pad_token_id=tokenizer.pad_token_id,
    )

    shifted_entropy = score_generated_entropy(
        generated_ids=generated_ids,
        generated_attention_mask=generated_attention_mask,
        model=model,
        score_micro_batch_size=score_micro_batch_size,
        device=device,
    )
    target_ids = generated_ids[:, 1:].cpu()
    target_mask = generated_attention_mask[:, 1:].bool().cpu()
    response_mask = build_response_mask(
        target_mask=target_mask,
        prompt_width=prompt_width,
        include_special_tokens=include_special_tokens,
        target_ids=target_ids,
        special_ids=set(tokenizer.all_special_ids),
    )

    results: list[ScoredPrompt] = []
    for row_idx, sample in enumerate(batch):
        valid_positions = response_mask[row_idx].nonzero(as_tuple=False).squeeze(1)
        token_ids: list[int] = []
        token_indices: list[int] = []
        entropy_values: list[torch.Tensor] = []

        for position in valid_positions.tolist():
            token_id = int(target_ids[row_idx, position].item())
            original_index = position + 1
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
                token_indices=token_indices,
                selected_mask=torch.zeros_like(token_entropy, dtype=torch.bool),
            )

        response_ids = generated_ids[row_idx, prompt_width:]
        response_attention_mask = generated_attention_mask[row_idx, prompt_width:].bool()
        generated_response = tokenizer.decode(
            response_ids[response_attention_mask],
            skip_special_tokens=True,
        )
        results.append(
            ScoredPrompt(
                sample=sample,
                model_prompt=prompt_texts[row_idx],
                generated_response=generated_response,
                token_scores=token_scores,
            )
        )

    return results


def score_prompt_response_batch(
    batch: list[DapoSample],
    prompt_texts: list[str],
    generated_responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    score_micro_batch_size: int,
    include_special_tokens: bool,
) -> list[ScoredPrompt]:
    prompt_token_ids = [
        tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
        for prompt_text in prompt_texts
    ]
    full_texts = [
        f"{prompt_text}{generated_response}"
        for prompt_text, generated_response in zip(prompt_texts, generated_responses, strict=True)
    ]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length + max_new_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    shifted_entropy = score_generated_entropy(
        generated_ids=encoded["input_ids"],
        generated_attention_mask=encoded["attention_mask"],
        model=model,
        score_micro_batch_size=score_micro_batch_size,
        device=device,
    )

    target_ids = encoded["input_ids"][:, 1:].cpu()
    target_mask = encoded["attention_mask"][:, 1:].bool().cpu()
    full_attention_mask = encoded["attention_mask"].cpu()
    special_ids = set(tokenizer.all_special_ids)

    results: list[ScoredPrompt] = []
    for row_idx, sample in enumerate(batch):
        full_len = int(full_attention_mask[row_idx].sum().item())
        pad_len = full_attention_mask.shape[1] - full_len
        response_start = pad_len + len(prompt_token_ids[row_idx])

        response_mask = build_variable_response_mask(
            target_mask=target_mask[row_idx],
            response_start=response_start,
            include_special_tokens=include_special_tokens,
            target_ids=target_ids[row_idx],
            special_ids=special_ids,
        )
        token_scores = collect_token_scores(
            response_mask=response_mask,
            target_ids=target_ids[row_idx],
            shifted_entropy=shifted_entropy[row_idx],
            tokenizer=tokenizer,
        )
        results.append(
            ScoredPrompt(
                sample=sample,
                model_prompt=prompt_texts[row_idx],
                generated_response=generated_responses[row_idx],
                token_scores=token_scores,
            )
        )

    return results


def generate_with_vllm(
    prompt_texts: list[str],
    model_name: str,
    base_url: str,
    timeout: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    """Generate responses with a vLLM OpenAI-compatible completions endpoint."""

    url = f"{base_url.rstrip('/')}/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt_texts,
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0,
        "top_p": top_p,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to vLLM at {url}: {error}") from error

    choices = body.get("choices")
    if not isinstance(choices, list):
        raise RuntimeError(f"Unexpected vLLM response: {body}")

    responses = [""] * len(prompt_texts)
    for fallback_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        index = int(choice.get("index", fallback_index))
        if 0 <= index < len(responses):
            responses[index] = str(choice.get("text", ""))
    return responses


def generate_and_score_with_vllm_logprobs(
    batch: list[DapoSample],
    prompt_texts: list[str],
    model_name: str,
    base_url: str,
    timeout: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    logprobs: int,
) -> list[ScoredPrompt]:
    """Generate responses and approximate token entropy from vLLM top logprobs."""

    url = f"{base_url.rstrip('/')}/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt_texts,
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0,
        "top_p": top_p,
        "logprobs": logprobs,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to vLLM at {url}: {error}") from error

    choices = body.get("choices")
    if not isinstance(choices, list):
        raise RuntimeError(f"Unexpected vLLM response: {body}")

    results: list[ScoredPrompt | None] = [None] * len(batch)
    for fallback_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        index = int(choice.get("index", fallback_index))
        if not 0 <= index < len(batch):
            continue

        generated_response = str(choice.get("text", ""))
        token_scores = token_scores_from_vllm_logprobs(choice.get("logprobs"))
        results[index] = ScoredPrompt(
            sample=batch[index],
            model_prompt=prompt_texts[index],
            generated_response=generated_response,
            token_scores=token_scores,
        )

    return [
        result
        if result is not None
        else ScoredPrompt(sample=batch[index], model_prompt=prompt_texts[index], generated_response="", token_scores=[])
        for index, result in enumerate(results)
    ]


def token_scores_from_vllm_logprobs(logprobs_payload: Any) -> list[TokenScore]:
    if not isinstance(logprobs_payload, dict):
        return []

    tokens = logprobs_payload.get("tokens") or []
    token_logprobs = logprobs_payload.get("token_logprobs") or []
    top_logprobs = logprobs_payload.get("top_logprobs") or []

    token_scores: list[TokenScore] = []
    for index, token in enumerate(tokens):
        if index >= len(token_logprobs):
            continue
        selected_token_logprob = token_logprobs[index]
        if selected_token_logprob is None:
            continue

        top_token_logprobs = top_logprobs[index] if index < len(top_logprobs) else None
        entropy = approximate_entropy_from_logprobs(top_token_logprobs, float(selected_token_logprob))
        token_scores.append(
            TokenScore(
                index=index,
                token_id=-1,
                token=str(token),
                entropy=entropy,
                selected=False,
            )
        )
    return token_scores


def approximate_entropy_from_logprobs(top_logprobs: Any, selected_token_logprob: float) -> float:
    """Approximate entropy from top-k logprobs, adding residual probability as one bucket."""

    logprob_values: list[float] = []
    if isinstance(top_logprobs, dict):
        logprob_values = [float(value) for value in top_logprobs.values() if value is not None]
    elif isinstance(top_logprobs, list):
        for item in top_logprobs:
            if isinstance(item, dict) and item.get("logprob") is not None:
                logprob_values.append(float(item["logprob"]))

    logprob_values.append(selected_token_logprob)
    unique_logprobs = list(dict.fromkeys(logprob_values))
    probs = [math.exp(logprob) for logprob in unique_logprobs]

    entropy = -sum(prob * math.log(max(prob, 1e-45)) for prob in probs)
    residual = max(0.0, 1.0 - sum(probs))
    if residual > 0:
        entropy -= residual * math.log(max(residual, 1e-45))
    return float(entropy)


def build_variable_response_mask(
    target_mask: torch.Tensor,
    response_start: int,
    include_special_tokens: bool,
    target_ids: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    """Build a response mask for variable prompt lengths after re-tokenization."""

    target_positions = torch.arange(target_mask.shape[0]) + 1
    response_mask = target_mask & (target_positions >= response_start)
    if include_special_tokens:
        return response_mask

    special_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for special_id in special_ids:
        special_mask |= target_ids == special_id
    return response_mask & ~special_mask


def collect_token_scores(
    response_mask: torch.Tensor,
    target_ids: torch.Tensor,
    shifted_entropy: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> list[TokenScore]:
    valid_positions = response_mask.nonzero(as_tuple=False).squeeze(1)
    token_ids: list[int] = []
    token_indices: list[int] = []
    entropy_values: list[torch.Tensor] = []

    for position in valid_positions.tolist():
        token_id = int(target_ids[position].item())
        original_index = position + 1
        token_ids.append(token_id)
        token_indices.append(original_index)
        entropy_values.append(shifted_entropy[position])

    if not entropy_values:
        return []

    token_entropy = torch.stack(entropy_values)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return build_token_scores(
        token_ids=token_ids,
        tokens=tokens,
        entropy=token_entropy,
        token_indices=token_indices,
        selected_mask=torch.zeros_like(token_entropy, dtype=torch.bool),
    )


def score_generated_entropy(
    generated_ids: torch.Tensor,
    generated_attention_mask: torch.Tensor,
    model: PreTrainedModel,
    score_micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute entropy in smaller chunks to avoid holding large batch logits."""

    entropy_chunks: list[torch.Tensor] = []
    for start in range(0, generated_ids.shape[0], score_micro_batch_size):
        end = start + score_micro_batch_size
        outputs = model(
            input_ids=generated_ids[start:end],
            attention_mask=generated_attention_mask[start:end],
            use_cache=False,
        )
        entropy_chunks.append(entropy_from_logits(outputs.logits[:, :-1, :]).cpu())
        del outputs
        release_cuda_memory(device)
    return torch.cat(entropy_chunks, dim=0)


def release_cuda_memory(device: torch.device) -> None:
    """Release temporary CUDA allocations between generation/scoring chunks."""

    gc.collect()
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def build_generated_attention_mask(
    generated_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_width: int,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Build an attention mask for prompt plus generated response tokens."""

    attention_mask = torch.ones_like(generated_ids, dtype=prompt_attention_mask.dtype)
    attention_mask[:, :prompt_width] = prompt_attention_mask
    if generated_ids.shape[1] > prompt_width and pad_token_id is not None:
        attention_mask[:, prompt_width:] = (generated_ids[:, prompt_width:] != pad_token_id).to(
            prompt_attention_mask.dtype
        )
    return attention_mask


def build_response_mask(
    target_mask: torch.Tensor,
    prompt_width: int,
    include_special_tokens: bool,
    target_ids: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    """Mask shifted target positions so only generated response tokens are scored."""

    target_positions = torch.arange(target_mask.shape[1]).unsqueeze(0) + 1
    response_mask = target_mask & (target_positions >= prompt_width)
    if include_special_tokens:
        return response_mask

    special_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for special_id in special_ids:
        special_mask |= target_ids == special_id
    return response_mask & ~special_mask


def apply_dataset_global_selection(scored_prompts: list[ScoredPrompt], importance_ratio: float) -> None:
    """Mark top-entropy tokens across every scored prompt in the run."""

    entropy_values = [
        token_score.entropy
        for scored_prompt in scored_prompts
        for token_score in scored_prompt.token_scores
    ]
    if not entropy_values:
        return

    flat_entropy = torch.tensor(entropy_values, dtype=torch.float32)
    flat_selected_mask = top_entropy_mask(flat_entropy, importance_ratio).tolist()

    flat_offset = 0
    for scored_prompt in scored_prompts:
        updated_scores: list[TokenScore] = []
        for token_score in scored_prompt.token_scores:
            updated_scores.append(replace(token_score, selected=bool(flat_selected_mask[flat_offset])))
            flat_offset += 1
        scored_prompt.token_scores = updated_scores


def render_prompt_for_model(
    sample: DapoSample,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = False,
) -> str:
    """Use the tokenizer chat template when possible, otherwise fall back to plain text."""

    if sample.prompt_messages is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                sample.prompt_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return sample.prompt_text
    return sample.prompt_text


def format_result(
    sample: DapoSample,
    token_scores: list[TokenScore],
    model_prompt: str,
    generated_response: str,
) -> dict[str, Any]:
    selected_scores = [score for score in token_scores if score.selected]

    return {
        "sample_id": sample.sample_id,
        "data_source": sample.data_source,
        "ground_truth": sample.ground_truth,
        "prompt_text": sample.prompt_text,
        "model_prompt": model_prompt,
        "generated_response": generated_response,
        "token_indices": [score.index for score in token_scores],
        "token_ids": [score.token_id for score in token_scores],
        "tokens": [score.token for score in token_scores],
        "entropy_scores": [score.entropy for score in token_scores],
        "selected_token_indices": [score.index for score in selected_scores],
        "selected_token_ids": [score.token_id for score in selected_scores],
        "selected_tokens": [score.token for score in selected_scores],
        "selected_entropy_scores": [score.entropy for score in selected_scores],
    }


def format_selected_tokens(scored_prompts: list[ScoredPrompt]) -> list[str]:
    selected_tokens: list[str] = []
    for scored_prompt in scored_prompts:
        for token_score in scored_prompt.token_scores:
            if not token_score.selected:
                continue
            selected_tokens.append(token_score.token)
    return selected_tokens


if __name__ == "__main__":
    main()
