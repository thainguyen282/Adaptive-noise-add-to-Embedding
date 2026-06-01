# Adaptive-noise-add-to-Embedding

Minimal proof-of-concept pipeline for finding important tokens in DAPO-style math prompts.

The first implementation defines important tokens as the highest-entropy prompt tokens under a causal language model. It is inspired by PPO actor entropy scoring, but it intentionally does not run reinforcement learning.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Run The PoC

```powershell
python -m dapo_token_importance.pipeline `
  --dataset sungyub/dapo-math-17k-verl `
  --split train `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --max-samples 32 `
  --batch-size 2 `
  --importance-ratio 0.2 `
  --output outputs/dapo_entropy_tokens.jsonl
```

For a quick smoke test, start with `--max-samples 2`.

## What The Pipeline Does

1. Loads a VERL-style DAPO dataset from Hugging Face.
2. Normalizes each chat prompt into text, using the tokenizer chat template when available.
3. Runs a causal LM forward pass over each prompt.
4. Computes token-level entropy from logits.
5. Selects the top `--importance-ratio` entropy tokens per sample.
6. Writes one JSON object per sample to JSONL.

## Output Schema

Each JSONL row contains:

- `sample_id`: dataset row index inside the loaded subset.
- `data_source`: original DAPO source metadata when available.
- `ground_truth`: reward-model answer when available.
- `prompt_text`: normalized human-readable prompt.
- `model_prompt`: exact prompt string passed to the tokenizer/model.
- `token_indices`, `token_ids`, `tokens`, `entropy_scores`: all scored prompt tokens.
- `selected_token_indices`, `selected_token_ids`, `selected_tokens`, `selected_entropy_scores`: top-entropy tokens.

By default, tokenizer special tokens are excluded from the ranking. Add `--include-special-tokens` if you want to inspect them too.

## Current Limitations

- Scores prompt tokens only. DAPO does not include generated model responses, so response-mask scoring can be added later after generation.
- Entropy is computed from the full vocabulary distribution, which can be memory-heavy for large models or long prompts.
- The first token is skipped because causal LM logits score the next token given previous context.