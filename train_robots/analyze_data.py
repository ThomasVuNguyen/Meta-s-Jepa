"""Quick analysis of the 3 multi-task datasets + models."""
import numpy as np
import torch
import json
from pathlib import Path

DATA = Path("train_robots/data")
MODELS = Path("train_robots/models/models")

print("=" * 70)
print("MULTI-TASK DATASET ANALYSIS")
print("=" * 70)

for name in ["reacher_easy_1k", "point_mass_easy_1k", "cartpole_swingup_1k"]:
    f = DATA / f"{name}.npz"
    if not f.exists():
        print(f"\n{name}: NOT FOUND"); continue
    
    d = np.load(str(f))
    z_t = d["z_t"]
    a_t = d["a_t"]
    z_next = d["z_next"]
    rewards = d["rewards"]
    
    n = len(z_t)
    n_eps = n // 200  # 200 steps per episode
    
    # Reshape rewards to episodes
    ep_rewards = rewards.reshape(n_eps, 200)
    ep_returns = ep_rewards.sum(axis=1)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Transitions:  {n:,}")
    print(f"  Episodes:     {n_eps}")
    print(f"  File size:    {f.stat().st_size/1e9:.2f} GB")
    print(f"")
    print(f"  z_t shape:    {z_t.shape}  (latent_dim={z_t.shape[1]})")
    print(f"  a_t shape:    {a_t.shape}  (action_dim={a_t.shape[1]})")
    print(f"")
    print(f"  --- Embedding Statistics ---")
    print(f"  z_t mean:     {z_t.mean():.4f}")
    print(f"  z_t std:      {z_t.std():.4f}")
    print(f"  z_t min/max:  [{z_t.min():.3f}, {z_t.max():.3f}]")
    print(f"  z_t norm mean: {np.linalg.norm(z_t, axis=1).mean():.2f}")
    print(f"")
    print(f"  --- Action Statistics ---")
    print(f"  a_t mean:     {a_t.mean(axis=0)}")
    print(f"  a_t std:      {a_t.std(axis=0)}")
    print(f"  a_t range:    [{a_t.min():.3f}, {a_t.max():.3f}]")
    print(f"")
    print(f"  --- Reward Statistics ---")
    print(f"  Per-step reward mean: {rewards.mean():.4f}")
    print(f"  Per-step reward std:  {rewards.std():.4f}")
    print(f"  Nonzero rewards:      {(rewards > 0).mean():.1%}")
    print(f"")
    print(f"  Episode return mean:  {ep_returns.mean():.2f}")
    print(f"  Episode return std:   {ep_returns.std():.2f}")
    print(f"  Episode return range: [{ep_returns.min():.1f}, {ep_returns.max():.1f}]")
    print(f"  Success rate (R>100): {(ep_returns > 100).mean():.1%}")
    
    # Check embedding diversity (cosine sim between random pairs)
    idx = np.random.choice(n, size=min(1000, n), replace=False)
    sample = z_t[idx]
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample_normed = sample / (norms + 1e-8)
    cos_sim = sample_normed @ sample_normed.T
    np.fill_diagonal(cos_sim, 0)
    avg_cos = cos_sim.sum() / (len(idx) * (len(idx) - 1))
    print(f"  Avg cosine sim (random pairs): {avg_cos:.4f}")
    
    # Check temporal coherence (consecutive embeddings should be similar)
    if n > 200:
        consec_cos = []
        for i in range(0, min(10000, n-1)):
            a = z_t[i] / (np.linalg.norm(z_t[i]) + 1e-8)
            b = z_next[i] / (np.linalg.norm(z_next[i]) + 1e-8)
            consec_cos.append(np.dot(a, b))
        print(f"  Avg cosine sim (consecutive):  {np.mean(consec_cos):.4f}")
        print(f"  Temporal coherence:            {'HIGH' if np.mean(consec_cos) > 0.95 else 'MODERATE' if np.mean(consec_cos) > 0.8 else 'LOW'}")

# Analyze models
print(f"\n{'='*70}")
print(f"MODEL ANALYSIS")
print(f"{'='*70}")

for task in ["reacher_easy", "point_mass_easy", "cartpole_swingup"]:
    task_dir = MODELS / task
    if not task_dir.exists():
        print(f"\n  {task}: NO MODELS"); continue
    
    dyn_files = sorted(task_dir.glob("dyn_*.pt"))
    reward_file = task_dir / "reward.pt"
    
    print(f"\n  {task}:")
    print(f"    Dynamics ensemble: {len(dyn_files)} models")
    
    # Check ensemble diversity (do models predict differently?)
    if dyn_files:
        total_params = sum(p.numel() for p in torch.load(str(dyn_files[0]), map_location="cpu", weights_only=True).values())
        print(f"    Params per model:  {total_params:,}")
        
        # Load all ensemble weights and check diversity
        weights = []
        for f in dyn_files:
            sd = torch.load(str(f), map_location="cpu", weights_only=True)
            w = torch.cat([v.flatten() for v in sd.values()])
            weights.append(w)
        
        # Pairwise cosine similarity between ensemble members
        cos_sims = []
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                cs = torch.cosine_similarity(weights[i].unsqueeze(0), weights[j].unsqueeze(0)).item()
                cos_sims.append(cs)
        print(f"    Ensemble weight cosine sim: {np.mean(cos_sims):.4f} (lower=more diverse)")
    
    if reward_file.exists():
        r_params = sum(p.numel() for p in torch.load(str(reward_file), map_location="cpu", weights_only=True).values())
        print(f"    Reward model:      {r_params:,} params")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
