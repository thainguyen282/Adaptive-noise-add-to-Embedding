import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import json

from scale_analysis import save_scale_analysis

def load_entropy(entropy_path, vocab_size, device):
    entropy_data = torch.load(entropy_path, map_location=device)
    if not isinstance(entropy_data, dict) or "entropy" not in entropy_data:
        raise ValueError("Entropy file must be the .pt dict saved by dapo_token_importance.pipeline")
    entropy = entropy_data["entropy"]
    count = entropy_data.get("count")
    entropy = torch.as_tensor(entropy, dtype=torch.float32, device=device).flatten()
    if count is not None:
        count = torch.as_tensor(count, device=device).flatten()
    if entropy.numel() < vocab_size:
        pad_size = vocab_size - entropy.numel()
        entropy = torch.cat([entropy, torch.zeros(pad_size, dtype=entropy.dtype, device=device)])
        if count is not None:
            count = torch.cat([count, torch.zeros(pad_size, dtype=count.dtype, device=device)])
    elif entropy.numel() > vocab_size:
        raise ValueError(f"Expected at most {vocab_size} entropy values, got {entropy.numel()}")
    if count is not None and count.numel() != vocab_size:
        raise ValueError(f"Expected {vocab_size} entropy counts, got {count.numel()}")
    return entropy, count

def entropy_threshold_from_top_ratio(entropy, count, top_ratio):
    seen = count > 0 if count is not None else torch.ones_like(entropy, dtype=torch.bool)
    seen_entropy = entropy[seen]
    if seen_entropy.numel() == 0:
        raise ValueError("No observed entropy values were found")
    top_k = max(1, int(seen_entropy.numel() * top_ratio + 0.9999))
    threshold = torch.topk(seen_entropy, k=top_k).values[-1].item()
    max_entropy = torch.max(seen_entropy).item()
    return threshold, max_entropy

def update(a,partition, iteration, value, emb):
    emb[a][(partition*iteration):((partition+1)*iteration)] = value

def make_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def keep_prob_before_scale(temp_chunk, temp, vocab_size, eps_i):
    dist_chunk = torch.cdist(temp_chunk, temp, p=1.0) / temp.size(1)
    dist_chunk = torch.exp(-dist_chunk)
    dist_chunk = dist_chunk / (torch.sum(dist_chunk, dim=1, keepdim=True) - 1)
    min_vect_chunk, _ = torch.min(dist_chunk, dim=1)
    beta_vect_chunk = eps_i + np.log(vocab_size - 1) + torch.log(min_vect_chunk)
    return torch.exp(beta_vect_chunk) / (vocab_size - 1 + torch.exp(beta_vect_chunk))

def compute_gap_stats(chunkEmb, N, k, eps_i, entropy, entropy_threshold, entropy_range, entropy_scale, chunk_size, device):
    old_mean = torch.zeros(N, dtype=torch.float32, device=device)
    old_std = torch.zeros(N, dtype=torch.float32, device=device)
    new_mean = torch.zeros(N, dtype=torch.float32, device=device)
    new_std = torch.zeros(N, dtype=torch.float32, device=device)
    gap_mean = torch.zeros(N, dtype=torch.float32, device=device)
    gap_std = torch.zeros(N, dtype=torch.float32, device=device)
    gap_min = torch.zeros(N, dtype=torch.float32, device=device)
    gap_max = torch.zeros(N, dtype=torch.float32, device=device)
    scale_mean = torch.zeros(N, dtype=torch.float32, device=device)

    for i in tqdm(range(0, N, chunk_size), desc="Gap stats"):
        col_indices = torch.arange(i, min(i + chunk_size, N), device=device)
        chunk_len = col_indices.numel()
        entropy_chunk = entropy[i:i + chunk_size]
        relu_entropy = torch.relu((entropy_chunk - entropy_threshold) / entropy_range)
        base_scale = 1 + entropy_scale * relu_entropy

        old_probs = torch.zeros(chunk_len, k, dtype=torch.float32, device=device)
        new_probs = torch.zeros(chunk_len, k, dtype=torch.float32, device=device)
        gaps = torch.zeros(chunk_len, k, dtype=torch.float32, device=device)
        scales = torch.zeros(chunk_len, k, dtype=torch.float32, device=device)

        for j in range(k):
            temp = chunkEmb[j]
            temp_chunk = temp[i:i + chunk_size]
            keep_prob = keep_prob_before_scale(temp_chunk, temp, N, eps_i)
            scale = torch.minimum(base_scale, 1 / keep_prob)
            new_prob = keep_prob * scale
            old_probs[:, j] = keep_prob
            new_probs[:, j] = new_prob
            gaps[:, j] = new_prob - keep_prob
            scales[:, j] = scale

        old_mean[col_indices] = old_probs.mean(dim=1)
        old_std[col_indices] = old_probs.std(dim=1, unbiased=False)
        new_mean[col_indices] = new_probs.mean(dim=1)
        new_std[col_indices] = new_probs.std(dim=1, unbiased=False)
        gap_mean[col_indices] = gaps.mean(dim=1)
        gap_std[col_indices] = gaps.std(dim=1, unbiased=False)
        gap_min[col_indices] = gaps.min(dim=1).values
        gap_max[col_indices] = gaps.max(dim=1).values
        scale_mean[col_indices] = scales.mean(dim=1)

    return {
        "old_mean": old_mean,
        "old_std": old_std,
        "new_mean": new_mean,
        "new_std": new_std,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "scale_mean": scale_mean,
        "num_embedding_values": k,
    }

def save_gap_analysis(gap_stats, entropy, entropy_count, entropy_threshold, entropy_scale, entropy_top_ratio, noise_gap_json_output, noise_gap_pt_output, device):
    gap_mean = gap_stats["gap_mean"]
    changed = gap_mean > 0
    token_ids = torch.arange(gap_mean.numel(), device=device)

    if noise_gap_pt_output is not None:
        make_parent_dir(noise_gap_pt_output)
        torch.save(
            {
                "token_ids": token_ids.detach().cpu(),
                "old_prob_mean": gap_stats["old_mean"].detach().cpu(),
                "old_prob_std": gap_stats["old_std"].detach().cpu(),
                "new_prob_mean": gap_stats["new_mean"].detach().cpu(),
                "new_prob_std": gap_stats["new_std"].detach().cpu(),
                "gap_mean": gap_mean.detach().cpu(),
                "gap_std": gap_stats["gap_std"].detach().cpu(),
                "gap_min": gap_stats["gap_min"].detach().cpu(),
                "gap_max": gap_stats["gap_max"].detach().cpu(),
                "scale_mean": gap_stats["scale_mean"].detach().cpu(),
                "num_embedding_values": torch.full((gap_mean.numel(),), gap_stats["num_embedding_values"], dtype=torch.long),
            },
            noise_gap_pt_output,
        )
        print(f"Saved noise gap tensor stats to {noise_gap_pt_output}")

    if noise_gap_json_output is not None:
        make_parent_dir(noise_gap_json_output)
        changed_indices = torch.nonzero(changed, as_tuple=True)[0]
        top_count = min(200, changed_indices.numel())
        top_local = changed_indices[torch.topk(gap_mean[changed], k=top_count).indices] if top_count > 0 else []
        token_summaries = []
        for token_id in top_local:
            token_id = int(token_id.item())
            token_summaries.append(
                {
                    "token_id": token_id,
                    "entropy": float(entropy[token_id].item()) if entropy is not None else None,
                    "entropy_count": int(entropy_count[token_id].item()) if entropy_count is not None else None,
                    "scale_mean": float(gap_stats["scale_mean"][token_id].item()),
                    "old_prob_mean": float(gap_stats["old_mean"][token_id].item()),
                    "old_prob_std": float(gap_stats["old_std"][token_id].item()),
                    "new_prob_mean": float(gap_stats["new_mean"][token_id].item()),
                    "new_prob_std": float(gap_stats["new_std"][token_id].item()),
                    "gap_mean": float(gap_mean[token_id].item()),
                    "gap_std": float(gap_stats["gap_std"][token_id].item()),
                    "gap_min": float(gap_stats["gap_min"][token_id].item()),
                    "gap_max": float(gap_stats["gap_max"][token_id].item()),
                    "relative_gain_mean": float((gap_stats["new_mean"][token_id] / gap_stats["old_mean"][token_id].clamp_min(1e-12)).item()),
                    "num_embedding_values": gap_stats["num_embedding_values"],
                }
            )

        summary = {
            "entropy_threshold": float(entropy_threshold),
            "entropy_scale": float(entropy_scale),
            "entropy_top_ratio": float(entropy_top_ratio),
            "tokens_seen": int(gap_mean.numel()),
            "tokens_changed": int(changed.sum().item()),
            "gap_mean_global": float(gap_mean.mean().item()),
            "gap_std_global": float(gap_mean.std(unbiased=False).item()),
            "gap_min_global": float(gap_stats["gap_min"].min().item()),
            "gap_max_global": float(gap_stats["gap_max"].max().item()),
            "top_tokens_by_gap_mean": token_summaries,
        }
        with open(noise_gap_json_output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
        print(f"Saved noise gap JSON summary to {noise_gap_json_output}")

def main(eps,iteration,adaptive_noise,entropy_path,entropy_threshold,entropy_scale,entropy_top_ratio,scale_plot_output,scale_json_output,show_scale_plot,noise_gap_json_output,noise_gap_pt_output):
    k = 1024
    eps_i = eps/k
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # original = torch.load("/project/phan/codellama/Tensor/Gemma/original.pt").to(device).to(torch.float32)
    original = torch.load("/project/phan/codellama/Tensor/Qwen3.5-9B/original.pt").to(device).to(torch.float32)
    # original = torch.load(f"/project/phan/codellama/Tensor/Qwen3-235B/Qwen3-235B-eps27.pt").to(device).to(torch.float32)
    print(original.size())

    # with torch.no_grad():
    #     emb = original.clone()
    emb = original.clone()
    
    
    # save_dir = f"/project/phan/codellama/Tensor/Gemma/Gemma-9B-eps27.pt"
    save_dir = f"/project/phan/tqn/Adaptive-noise-add-to-Embedding/Tensor/Qwen3.5-9B/Qwen3.5-9B-eps{eps}.pt"
    print(save_dir)
    chunkEmb = torch.chunk(emb, k, dim=1)
    N = original.size(0)
    entropy = None
    entropy_count = None
    gap_stats_enabled = adaptive_noise and (noise_gap_json_output is not None or noise_gap_pt_output is not None)
    if adaptive_noise:
        if entropy_path is None:
            raise ValueError("--entropy_path is required when --adaptive_noise is enabled")
        entropy, entropy_count = load_entropy(entropy_path, N, device)
        if entropy_threshold is None:
            entropy_threshold, max_entropy = entropy_threshold_from_top_ratio(entropy, entropy_count, entropy_top_ratio)
        else:
            seen = entropy_count > 0 if entropy_count is not None else torch.ones_like(entropy, dtype=torch.bool)
            max_entropy = torch.max(entropy[seen]).item()
        entropy_range = max(max_entropy - entropy_threshold, 1e-12)
        save_scale_analysis(entropy, entropy_count, entropy_threshold, entropy_range, entropy_scale, entropy_top_ratio, scale_plot_output, scale_json_output, show_scale_plot)

    if gap_stats_enabled:
        old_sum = torch.zeros(N, dtype=torch.float32, device=device)
        old_sq_sum = torch.zeros(N, dtype=torch.float32, device=device)
        new_sum = torch.zeros(N, dtype=torch.float32, device=device)
        new_sq_sum = torch.zeros(N, dtype=torch.float32, device=device)
        gap_sum = torch.zeros(N, dtype=torch.float32, device=device)
        gap_sq_sum = torch.zeros(N, dtype=torch.float32, device=device)
        gap_min = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
        gap_max = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
        scale_sum = torch.zeros(N, dtype=torch.float32, device=device)
        stats_count = torch.zeros(N, dtype=torch.long, device=device)

    chunk_size = 20000
    # chunk_size = 256000
    dist_list = []
    keep_prob_list = []
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    for it in tqdm(range(iteration-1, 1024)):
        print("Iteration", it, "Eps: ", eps)
        temp = chunkEmb[iteration]
        for i in tqdm(range(0, N, chunk_size)):
            temp_chunk = temp[i:i+chunk_size]  # [chunk_size, D]
            # print(temp_chunk.size())
            # print(temp.size())
            dist_chunk = torch.cdist(temp_chunk, temp, p=1.0) / temp.size(1)
            dist_chunk = torch.exp(-dist_chunk)
            dist_chunk = dist_chunk / (torch.sum(dist_chunk, dim=1, keepdim=True) - 1)
            min_vect_chunk, _ = torch.min(dist_chunk, dim=1)
            beta_vect_chunk = eps_i + np.log(original.size(0)-1) + torch.log(min_vect_chunk)
            
            
            keep_prob_chunk = torch.exp(beta_vect_chunk)/(original.size(0)-1+torch.exp(beta_vect_chunk))
            dist_chunk = dist_chunk*((original.size(0) - 1) / (original.size(0) - 1 + torch.exp(beta_vect_chunk))).unsqueeze(1)
            keep_prob_before_scale = keep_prob_chunk.flatten()
            # plot mean and std of keep_prob_before_scale
            print(f"Mean of keep_prob_before_scale: {keep_prob_before_scale.mean().item()}")
            print(f"Std of keep_prob_before_scale: {keep_prob_before_scale.std().item()}")

            # create scale value the increase the probability of keep based on entropy
            if adaptive_noise:
                entropy_chunk = entropy[i:i+chunk_size]
                relu_entropy = torch.relu((entropy_chunk - entropy_threshold) / entropy_range)
                scale = 1 + entropy_scale * relu_entropy
                scale = torch.minimum(scale, 1 / keep_prob_before_scale)
            else:
                scale = 1
            keep_prob_chunk = keep_prob_before_scale*scale
            col_indices = torch.arange(i, min(i+chunk_size, N), device=temp.device)
            if gap_stats_enabled:
                gap = keep_prob_chunk - keep_prob_before_scale
                old_sum[col_indices] += keep_prob_before_scale
                old_sq_sum[col_indices] += keep_prob_before_scale ** 2
                new_sum[col_indices] += keep_prob_chunk
                new_sq_sum[col_indices] += keep_prob_chunk ** 2
                gap_sum[col_indices] += gap
                gap_sq_sum[col_indices] += gap ** 2
                gap_min[col_indices] = torch.minimum(gap_min[col_indices], gap)
                gap_max[col_indices] = torch.maximum(gap_max[col_indices], gap)
                scale_sum[col_indices] += scale
                stats_count[col_indices] += 1
            balance  = (1 - keep_prob_chunk.flatten()) / (1 - (keep_prob_chunk.flatten() / scale))
            dist_chunk = dist_chunk*balance.unsqueeze(1)

            row_indices = torch.arange(len(dist_chunk), device=temp.device)
            dist_chunk[row_indices, col_indices] = keep_prob_chunk

            cum_dist = torch.sum(dist_chunk, dim=1)
            
            check = torch.isclose(cum_dist, torch.ones_like(cum_dist), atol=1e-6)
            
            if torch.all(check):
                print("All rows sum up to 1")
            else:
                print("Some rows do NOT sum up to 1")
                print("Indices with error:", torch.nonzero(~check, as_tuple=True)[0])
                print("Problematic sums:", cum_dist[~check])
            
            
            indice_chunk = torch.multinomial(dist_chunk, 1)
            rows_to_update = torch.arange(i, min(i+chunk_size, N), device=temp.device)
            original[rows_to_update, iteration] = emb[indice_chunk.flatten(), iteration]
            del dist_chunk, keep_prob_chunk
            torch.cuda.empty_cache()
        
        
        if it % 100 == 0 or it == 1023: 
            print(f"reach step: {it+1}/1023")
            torch.save(original,save_dir)

    if gap_stats_enabled:
        seen = stats_count > 0
        count_float = stats_count[seen].float()
        old_mean = old_sum[seen] / count_float
        new_mean = new_sum[seen] / count_float
        gap_mean = gap_sum[seen] / count_float
        old_std = torch.sqrt(torch.clamp(old_sq_sum[seen] / count_float - old_mean ** 2, min=0))
        new_std = torch.sqrt(torch.clamp(new_sq_sum[seen] / count_float - new_mean ** 2, min=0))
        gap_std = torch.sqrt(torch.clamp(gap_sq_sum[seen] / count_float - gap_mean ** 2, min=0))
        scale_mean = scale_sum[seen] / count_float
        token_ids = torch.arange(N, device=device)[seen]
        changed = gap_mean > 0

        if noise_gap_pt_output is not None:
            make_parent_dir(noise_gap_pt_output)
            torch.save(
                {
                    "token_ids": token_ids.detach().cpu(),
                    "old_prob_mean": old_mean.detach().cpu(),
                    "old_prob_std": old_std.detach().cpu(),
                    "new_prob_mean": new_mean.detach().cpu(),
                    "new_prob_std": new_std.detach().cpu(),
                    "gap_mean": gap_mean.detach().cpu(),
                    "gap_std": gap_std.detach().cpu(),
                    "gap_min": gap_min[seen].detach().cpu(),
                    "gap_max": gap_max[seen].detach().cpu(),
                    "scale_mean": scale_mean.detach().cpu(),
                    "num_embedding_values": stats_count[seen].detach().cpu(),
                },
                noise_gap_pt_output,
            )
            print(f"Saved noise gap tensor stats to {noise_gap_pt_output}")

        if noise_gap_json_output is not None:
            make_parent_dir(noise_gap_json_output)
            changed_indices = torch.nonzero(changed, as_tuple=True)[0]
            top_count = min(200, changed_indices.numel())
            top_local = changed_indices[torch.topk(gap_mean[changed], k=top_count).indices] if top_count > 0 else []
            token_summaries = []
            for idx in top_local:
                token_id = int(token_ids[idx].item())
                token_summaries.append(
                    {
                        "token_id": token_id,
                        "entropy": float(entropy[token_id].item()) if entropy is not None else None,
                        "entropy_count": int(entropy_count[token_id].item()) if entropy_count is not None else None,
                        "scale_mean": float(scale_mean[idx].item()),
                        "old_prob_mean": float(old_mean[idx].item()),
                        "old_prob_std": float(old_std[idx].item()),
                        "new_prob_mean": float(new_mean[idx].item()),
                        "new_prob_std": float(new_std[idx].item()),
                        "gap_mean": float(gap_mean[idx].item()),
                        "gap_std": float(gap_std[idx].item()),
                        "gap_min": float(gap_min[token_id].item()),
                        "gap_max": float(gap_max[token_id].item()),
                        "relative_gain_mean": float((new_mean[idx] / old_mean[idx].clamp_min(1e-12)).item()),
                        "num_embedding_values": int(stats_count[token_id].item()),
                    }
                )

            summary = {
                "entropy_threshold": float(entropy_threshold),
                "entropy_scale": float(entropy_scale),
                "entropy_top_ratio": float(entropy_top_ratio),
                "tokens_seen": int(seen.sum().item()),
                "tokens_changed": int(changed.sum().item()),
                "gap_mean_global": float(gap_mean.mean().item()),
                "gap_std_global": float(gap_mean.std(unbiased=False).item()),
                "gap_min_global": float(gap_min[seen].min().item()),
                "gap_max_global": float(gap_max[seen].max().item()),
                "top_tokens_by_gap_mean": token_summaries,
            }
            with open(noise_gap_json_output, "w", encoding="utf-8") as output_file:
                json.dump(summary, output_file, indent=2)
            print(f"Saved noise gap JSON summary to {noise_gap_json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--eps', type=float, required=True, help='Epsilon value')
    parser.add_argument('--i', type=int, required=True, help='Iteration')
    parser.add_argument('--adaptive_noise', type=bool, required=True, help='Adaptive noise')
    parser.add_argument('--entropy_path', type=str, default=None, help='Path to entropy tensor for adaptive noise')
    parser.add_argument('--entropy_threshold', type=float, default=None, help='Entropy threshold where keep-prob scaling starts')
    parser.add_argument('--entropy_scale', type=float, default=1.0, help='Maximum added scale for high-entropy tokens')
    parser.add_argument('--entropy_top_ratio', type=float, default=0.1, help='Top entropy ratio to scale when threshold is not set')
    parser.add_argument('--scale_plot_output', type=str, default=None, help='Optional path to save scale comparison plot')
    parser.add_argument('--scale_json_output', type=str, default=None, help='Optional path to save scale comparison JSON')
    parser.add_argument('--show_scale_plot', action='store_true', help='Show scale comparison plot interactively')
    parser.add_argument('--noise_gap_json_output', type=str, default=None, help='Optional path to save keep-probability gap summary JSON')
    parser.add_argument('--noise_gap_pt_output', type=str, default=None, help='Optional path to save keep-probability gap tensors')

    args = parser.parse_args()
    main(args.eps,args.i,args.adaptive_noise,args.entropy_path,args.entropy_threshold,args.entropy_scale,args.entropy_top_ratio,args.scale_plot_output,args.scale_json_output,args.show_scale_plot,args.noise_gap_json_output,args.noise_gap_pt_output)
              
