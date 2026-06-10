import json
import os

import torch


def save_scale_analysis(entropy, count, threshold, entropy_range, entropy_scale, entropy_top_ratio, plot_output=None, json_output=None, show_plot=False):
    seen = count > 0 if count is not None else torch.ones_like(entropy, dtype=torch.bool)
    scale = torch.ones_like(entropy)
    scale[seen] = 1 + entropy_scale * torch.relu((entropy[seen] - threshold) / entropy_range)
    difference = scale - 1

    if plot_output is not None or show_plot:
        import matplotlib.pyplot as plt

        sorted_scale = scale[seen][torch.argsort(entropy[seen])].detach().cpu()
        x = torch.arange(sorted_scale.numel())
        top_k = max(1, int(sorted_scale.numel() * entropy_top_ratio + 0.9999))
        plt.figure(figsize=(10, 5))
        plt.plot(x, torch.ones_like(x, dtype=torch.float32), label="no scale")
        plt.plot(x, sorted_scale, label="adaptive scale")
        plt.axvline(sorted_scale.numel() - top_k, color="red", linestyle="--", label="top-ratio threshold")
        plt.xlabel("Observed tokens sorted by entropy")
        plt.ylabel("Scale")
        plt.title("Adaptive Scale vs No Scale")
        plt.legend()
        plt.tight_layout()
        if plot_output is not None:
            os.makedirs(os.path.dirname(plot_output), exist_ok=True)
            plt.savefig(plot_output, dpi=200)
        if show_plot:
            plt.show()

    if json_output is not None:
        os.makedirs(os.path.dirname(json_output), exist_ok=True)
        summary = {
            "observed_tokens": int(seen.sum().item()),
            "unseen_tokens": int((~seen).sum().item()),
            "entropy_top_ratio": entropy_top_ratio,
            "entropy_scale": entropy_scale,
            "threshold": float(threshold),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "scale_mean_observed": float(scale[seen].mean().item()),
            "changed_tokens": int((difference > 0).sum().item()),
            "unchanged_tokens": int((difference == 0).sum().item()),
        }
        with open(json_output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
