import random
from copy import deepcopy
import torch
import numpy as np
import re
import pickle
import argparse
from tqdm import tqdm

def update(a,partition, iteration, value, emb):
    emb[a][(partition*iteration):((partition+1)*iteration)] = value

def main(eps,iteration):
    k = 1024
    eps_i = eps/k
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # original = torch.load("/project/phan/codellama/Tensor/Gemma/original.pt").to(device).to(torch.float32)
    original = torch.load("/project/phan/codellama/Tensor/Qwen3.5-0.8B/original.pt").to(device).to(torch.float32)
    # original = torch.load(f"/project/phan/codellama/Tensor/Qwen3-235B/Qwen3-235B-eps27.pt").to(device).to(torch.float32)
    print(original.size())

    # with torch.no_grad():
    #     emb = original.clone()
    emb = original.clone()
    
    
    # save_dir = f"/project/phan/codellama/Tensor/Gemma/Gemma-9B-eps27.pt"
    save_dir = f"/project/phan/codellama/Tensor/Qwen3.5-0.8B/Qwen3.5-0.8B-eps{eps}.pt"
    print(save_dir)
    chunkEmb = torch.chunk(emb, k, dim=1)
    N = original.size(0)
    chunk_size = 20000
    # chunk_size = 256000
    dist_list = []
    keep_prob_list = []
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
            dist_chunk[range(len(dist_chunk)), range(len(dist_chunk))] = keep_prob_chunk.flatten()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--eps', type=float, required=True, help='Epsilon value')
    parser.add_argument('--i', type=int, required=True, help='Iteration')

    args = parser.parse_args()
    main(args.eps,args.i)
              
