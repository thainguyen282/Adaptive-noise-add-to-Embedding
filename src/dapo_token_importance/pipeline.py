import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from data import load_data

from dapo_token_importance.data import DEFAULT_DATASET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect average token entropy with vLLM prompt logprobs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--entropy-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--vllm-logprobs", type=int, default=20)
    parser.add_argument("--avg-entropy-output", default="outputs/avg_token_entropy.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-every-batches", type=int, default=50)
    parser.add_argument("--include-special-tokens", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.entropy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(model=args.entropy_model, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=args.vllm_logprobs)

    dataset = load_data(args.dataset, split=args.split, verification_mode="no_checks")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path = Path(args.avg_entropy_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entropy_sum = torch.zeros(len(tokenizer), dtype=torch.float32)
    entropy_count = torch.zeros(len(tokenizer), dtype=torch.long)
    special_ids = set(tokenizer.all_special_ids)

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Scoring prompts"):
        prompts = []
        end = min(start + args.batch_size, len(dataset))
        for index in range(start, end):
            prompt = dataset[index].get("prompt")
            if isinstance(prompt, list) and getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            elif isinstance(prompt, list):
                prompt = "\n".join(f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in prompt)
            else:
                prompt = str(prompt)
            prompt_ids = tokenizer(prompt, truncation=True, max_length=args.max_length)["input_ids"]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs or []
            for top_logprobs in prompt_logprobs:
                if not top_logprobs:
                    continue
                token_id = None
                logprob_values = []
                for raw_id, item in top_logprobs.items():
                    logprob = item.logprob
                    rank = getattr(item, "rank", None)
                    logprob_values.append(float(logprob))
                    if rank == 1:
                        token_id = int(raw_id)
                if token_id is None:
                    token_id = int(max(top_logprobs, key=lambda key: top_logprobs[key].logprob))
                if token_id >= len(tokenizer) or (not args.include_special_tokens and token_id in special_ids):
                    continue
                probs = torch.tensor([math.exp(logprob) for logprob in logprob_values], dtype=torch.float32)
                entropy = -(probs * torch.log(probs.clamp_min(1e-45))).sum()
                entropy_sum[token_id] += entropy
                entropy_count[token_id] += 1

        if args.save_every_batches and (start // args.batch_size + 1) % args.save_every_batches == 0:
            avg_entropy = torch.zeros_like(entropy_sum)
            seen = entropy_count > 0
            avg_entropy[seen] = entropy_sum[seen] / entropy_count[seen].float()
            torch.save({"entropy": avg_entropy, "sum": entropy_sum, "count": entropy_count, "processed_samples": end}, output_path)

    avg_entropy = torch.zeros_like(entropy_sum)
    seen = entropy_count > 0
    avg_entropy[seen] = entropy_sum[seen] / entropy_count[seen].float()
    torch.save({"entropy": avg_entropy, "sum": entropy_sum, "count": entropy_count, "processed_samples": len(dataset)}, output_path)
    print(f"Saved average token entropy to {output_path}")
