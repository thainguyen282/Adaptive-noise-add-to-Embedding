import argparse
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer


def load_gap_stats(gap_pt_path):
    data = torch.load(gap_pt_path, map_location="cpu")
    return {
        "token_ids": data["token_ids"].long(),
        "gap_mean": data["gap_mean"].float(),
        "gap_std": data.get("gap_std"),
        "old_prob_mean": data.get("old_prob_mean"),
        "new_prob_mean": data.get("new_prob_mean"),
        "scale_mean": data.get("scale_mean"),
    }


def load_entropy_for_tokens(entropy_path, token_ids):
    data = torch.load(entropy_path, map_location="cpu")
    entropy = data["entropy"].float().flatten()
    max_token_id = int(token_ids.max().item())
    if entropy.numel() <= max_token_id:
        pad_size = max_token_id + 1 - entropy.numel()
        entropy = torch.cat([entropy, torch.zeros(pad_size, dtype=entropy.dtype)])
    return entropy[token_ids]


def probability_gap(stats):
    old_prob_mean = stats.get("old_prob_mean")
    new_prob_mean = stats.get("new_prob_mean")
    if old_prob_mean is not None and new_prob_mean is not None:
        return new_prob_mean - old_prob_mean
    return stats["gap_mean"]


def decode_token_labels(tokenizer, token_ids):
    labels = []
    for token_id in token_ids:
        token_id = int(token_id.item())
        token = tokenizer.convert_ids_to_tokens(token_id)
        label = token.replace("Ġ", " ").replace("▁", " ").strip()
        if not label:
            label = tokenizer.decode([token_id])
        if len(label) > 18:
            label = label[:15] + "..."
        labels.append(label)
    return labels


def plot_noise_gap(gap_pt_path, output_dir, entropy_path=None, top_k=20, show_plot=False, tokenizer_model=None):
    os.makedirs(output_dir, exist_ok=True)
    stats = load_gap_stats(gap_pt_path)
    gap_mean = probability_gap(stats)
    token_ids = stats["token_ids"]

    plt.figure(figsize=(8, 5))
    plt.hist(gap_mean.numpy(), bins=50)
    plt.xlabel("Keep-probability gap (new - old)")
    plt.ylabel("Number of tokens")
    plt.title("Distribution of keep-probability gap")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "gap_histogram.png")
    plt.savefig(hist_path, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()

    bar_path = None
    changed = gap_mean > 0
    if changed.any():
        top_count = min(top_k, int(changed.sum().item()))
        top_values, top_idx = torch.topk(gap_mean[changed], k=top_count)
        top_token_ids = token_ids[changed][top_idx]
        if tokenizer_model is None:
            raise ValueError("tokenizer_model is required to label the bar chart with token text")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        labels = decode_token_labels(tokenizer, top_token_ids)

        plt.figure(figsize=(12, 5))
        plt.bar(range(top_count), top_values.numpy())
        plt.xticks(range(top_count), labels, rotation=45, ha="right")
        plt.xlabel("Token")
        plt.ylabel("Keep-probability gap (new - old)")
        plt.title(f"Top {top_count} tokens by keep-probability gap")
        plt.tight_layout()
        bar_path = os.path.join(output_dir, "top_gap_bar.png")
        plt.savefig(bar_path, dpi=200)
        if show_plot:
            plt.show()
        else:
            plt.close()

    scatter_path = None
    if entropy_path is not None:
        entropy_values = load_entropy_for_tokens(entropy_path, token_ids)
        plt.figure(figsize=(8, 5))
        plt.scatter(entropy_values.numpy(), gap_mean.numpy(), s=8, alpha=0.4)
        plt.xlabel("Average token entropy")
        plt.ylabel("Keep-probability gap mean")
        plt.title("Entropy vs keep-probability gap")
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, "entropy_vs_gap_scatter.png")
        plt.savefig(scatter_path, dpi=200)
        if show_plot:
            plt.show()
        else:
            plt.close()

    return {
        "histogram": hist_path,
        "top_bar": bar_path,
        "scatter": scatter_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot keep-probability gap analysis.")
    parser.add_argument("--gap_pt_path", required=True, help="Path to noise_gap_stats.pt from getVocabDP.py")
    parser.add_argument("--output_dir", default="outputs/noise_gap_plots")
    parser.add_argument("--entropy_path", default=None, help="Optional avg_token_entropy.pt for scatter plot")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top tokens in bar chart")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Tokenizer model for decoding token ids to text")
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()

    paths = plot_noise_gap(
        gap_pt_path=args.gap_pt_path,
        output_dir=args.output_dir,
        entropy_path=args.entropy_path,
        top_k=args.top_k,
        show_plot=args.show_plot,
        tokenizer_model=args.model,
    )

    for name, path in paths.items():
        if path is not None:
            print(f"Saved {name} plot to {path}")
