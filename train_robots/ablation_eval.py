#!/usr/bin/env python3
"""Phase 4: Ablation studies for teach-by-showing agent.

IMPORTANT: Set rendering env vars before any dm_control imports.
"""
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

"""

Tests 4 conditions on reacher_easy + cartpole_swingup, 5 seeds each:
  1. FULL:       ensemble=5, β=2.0, α=5.0  (baseline from Phase 7)
  2. SINGLE:     ensemble=1, β=0,   α=5.0  (single model, no uncertainty)
  3. NO_UNCERT:  ensemble=5, β=0,   α=5.0  (ensemble but no penalty)
  4. NO_REWARD:  ensemble=5, β=2.0, α=0    (no reward prediction)

Each condition × 2 tasks × 5 seeds × 2 modes (faithful+shifted) = 80 episodes
Total: 4 × 80 = 320 episodes, ~25 min
"""

import os, sys, time, json, logging
import numpy as np
import torch
import torch.nn as nn

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
BASE = "/home/ubuntu/vjepa_mvp"
MODEL_DIR = f"{BASE}/models"
RESULTS_DIR = f"{BASE}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model definitions (same as teach_by_showing.py) ─────────────────
class DynamicsModel(nn.Module):
    def __init__(self, z_dim=1024, a_dim=2, h_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))

class RewardModel(nn.Module):
    def __init__(self, z_dim=1024, a_dim=2, h_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, 1),
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)

# ── V-JEPA 2 Encoder ────────────────────────────────────────────────
_encoder = None
_device = None

def get_encoder():
    global _encoder, _device
    if _encoder is not None:
        return _encoder, _device
    
    from transformers import AutoModel
    import torchvision.transforms as T
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
    ).to(_device).eval()
    
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    class Encoder:
        def __init__(self, model, preprocess, device):
            self.model = model
            self.preprocess = preprocess
            self.device = device
            
        @torch.no_grad()
        def encode_frames(self, frames):
            """frames: list of HWC uint8 numpy arrays → (N, 1024) tensor"""
            tensors = torch.stack([self.preprocess(f) for f in frames])
            # V-JEPA expects (B, T, C, H, W) - add temporal dim at position 1
            tensors = tensors.unsqueeze(1).to(self.device)
            # Batch to avoid OOM
            embeddings = []
            bs = 16
            for i in range(0, len(tensors), bs):
                batch = tensors[i:i+bs]
                out = self.model(batch)
                if hasattr(out, 'last_hidden_state'):
                    emb = out.last_hidden_state.mean(dim=1)
                elif isinstance(out, dict) and 'last_hidden_state' in out:
                    emb = out['last_hidden_state'].mean(dim=1)
                else:
                    emb = out.mean(dim=1) if len(out.shape) > 2 else out
                embeddings.append(emb)
            return torch.cat(embeddings, dim=0)
    
    _encoder = Encoder(model, preprocess, _device)
    log.info("V-JEPA 2 ready.")
    return _encoder, _device

# ── Load models ──────────────────────────────────────────────────────
def load_task_models(task, device, n_ensemble=5):
    """Load dynamics ensemble + reward model for a task."""
    task_dir = f"{MODEL_DIR}/{task}"
    
    # Detect action dim from saved model
    ckpt = torch.load(f"{task_dir}/dyn_0.pt", map_location=device, weights_only=True)
    first_weight = list(ckpt.values())[0]
    a_dim = first_weight.shape[1] - 1024
    
    dyn_models = []
    for i in range(n_ensemble):
        m = DynamicsModel(1024, a_dim).to(device)
        m.load_state_dict(torch.load(f"{task_dir}/dyn_{i}.pt", map_location=device, weights_only=True))
        m.eval()
        dyn_models.append(m)
    
    reward_model = RewardModel(1024, a_dim).to(device)
    reward_model.load_state_dict(torch.load(f"{task_dir}/reward.pt", map_location=device, weights_only=True))
    reward_model.eval()
    
    return dyn_models, reward_model, a_dim

# ── DMC Environment ─────────────────────────────────────────────────
def make_env(task, seed=0):
    from dm_control import suite
    domain, task_name = task.split("_", 1)
    if domain == "point" and task_name.startswith("mass"):
        domain = "point_mass"
        task_name = task_name[5:]  # "mass_easy" → "easy"
    env = suite.load(domain, task_name, task_kwargs={"random": seed})
    return env

def render_frame(env, size=224):
    return env.physics.render(height=size, width=size, camera_id=0)

def close_env(env):
    """Safely close a dm_control environment."""
    try:
        if hasattr(env, '_physics') and env._physics is not None:
            env._physics.free()
    except Exception:
        pass

def get_action_spec(env):
    spec = env.action_spec()
    return spec.shape[0], spec.minimum, spec.maximum

def run_expert_episode(env, n_steps=200):
    """Run heuristic expert, return frames + actions + rewards."""
    from dm_control import suite
    frames, actions, rewards = [], [], []
    ts = env.reset()
    frames.append(render_frame(env))
    
    for _ in range(n_steps):
        a_dim = env.action_spec().shape[0]
        action = np.random.uniform(env.action_spec().minimum, 
                                    env.action_spec().maximum, 
                                    size=a_dim)
        ts = env.step(action)
        frames.append(render_frame(env))
        actions.append(action)
        rewards.append(ts.reward or 0.0)
    
    return frames, actions, rewards

# ── CEM Planner ──────────────────────────────────────────────────────
@torch.no_grad()
def cem_plan(z_curr, goal_z_seq, dyn_models, reward_model,
             a_dim, a_min, a_max, device,
             n_candidates=500, n_elite=50, n_iters=5, horizon=8,
             alpha=5.0, beta=2.0):
    """CEM planning with configurable alpha (reward weight) and beta (uncertainty)."""
    
    mu = torch.zeros(horizon, a_dim, device=device)
    std = torch.ones(horizon, a_dim, device=device) * 0.5
    
    a_min_t = torch.tensor(a_min, device=device, dtype=torch.float32)
    a_max_t = torch.tensor(a_max, device=device, dtype=torch.float32)
    
    for iteration in range(n_iters):
        # Sample action sequences
        noise = torch.randn(n_candidates, horizon, a_dim, device=device)
        actions = mu.unsqueeze(0) + std.unsqueeze(0) * noise
        actions = actions.clamp(a_min_t, a_max_t)
        
        # Evaluate with ensemble
        scores = torch.zeros(n_candidates, device=device)
        z = z_curr.unsqueeze(0).expand(n_candidates, -1)
        
        for t in range(horizon):
            a = actions[:, t]
            
            # Ensemble predictions
            preds = torch.stack([m(z, a) for m in dyn_models])  # (E, N, z_dim)
            z_next = preds.mean(0)  # (N, z_dim)
            
            # Ensemble disagreement
            if beta > 0 and len(dyn_models) > 1:
                disagreement = preds.std(0).mean(-1)  # (N,)
                scores -= beta * disagreement
            
            # Reward prediction
            if alpha > 0:
                r_pred = reward_model(z, a)
                scores += alpha * r_pred
            
            # Goal distance
            goal_idx = min(t, len(goal_z_seq) - 1)
            goal_dist = (z_next - goal_z_seq[goal_idx]).pow(2).sum(-1).sqrt()
            scores -= goal_dist
            
            z = z_next
        
        # Select elites
        elite_idx = scores.topk(n_elite).indices
        elite_actions = actions[elite_idx]
        mu = elite_actions.mean(0)
        std = elite_actions.std(0).clamp(min=0.05)
    
    return mu[0].cpu().numpy()

# ── Evaluation Loop ──────────────────────────────────────────────────
def evaluate_condition(task, condition_name, n_ensemble, alpha, beta, 
                       n_demos=5, n_steps=200, lookahead=15):
    """Run one ablation condition on one task."""
    encoder, device = get_encoder()
    dyn_models_all, reward_model, a_dim = load_task_models(task, device, n_ensemble=5)
    
    # Select subset of models for this condition
    dyn_models = dyn_models_all[:n_ensemble]
    
    results = []
    
    for demo_idx in range(n_demos):
        seed = demo_idx
        
        # Record demo
        env = make_env(task, seed=seed)
        log.info(f"  Demo {demo_idx+1}/{n_demos} (seed={seed})...")
        frames, actions, rewards = run_expert_episode(env, n_steps)
        demo_return = sum(rewards)
        
        close_env(env)
        
        # Encode demo
        goal_embeddings = encoder.encode_frames(frames)
        
        for mode, replay_seed in [("faithful", seed), ("shifted", seed + 1000)]:
            env = make_env(task, seed=replay_seed)
            ts = env.reset()
            frame = render_frame(env)
            z_curr = encoder.encode_frames([frame])[0]
            
            a_spec = env.action_spec()
            a_min, a_max = a_spec.minimum, a_spec.maximum
            
            total_reward = 0.0
            progress = 0
            
            for step in range(n_steps):
                # Get goal subsequence
                start_idx = min(step, len(goal_embeddings) - lookahead)
                end_idx = min(start_idx + lookahead, len(goal_embeddings))
                goal_seq = goal_embeddings[start_idx:end_idx]
                
                action = cem_plan(
                    z_curr, goal_seq, dyn_models, reward_model,
                    a_dim, a_min, a_max, device,
                    alpha=alpha, beta=beta,
                )
                
                ts = env.step(action)
                total_reward += (ts.reward or 0.0)
                progress = step + 1
                
                frame = render_frame(env)
                z_curr = encoder.encode_frames([frame])[0]
            
            results.append({
                "task": task,
                "condition": condition_name,
                "mode": mode,
                "demo_seed": seed,
                "replay_seed": replay_seed,
                "demo_return": round(demo_return, 1),
                "replay_return": round(total_reward, 1),
                "progress": f"{progress}/{n_steps}",
                "n_ensemble": n_ensemble,
                "alpha": alpha,
                "beta": beta,
            })
            
            log.info(f"  {condition_name:12s} | {task:18s} | {mode:8s} | "
                     f"demo={demo_return:6.1f} → replay={total_reward:6.1f}")
            
            close_env(env)
    
    return results

# ── Main ─────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 4: ABLATION STUDIES")
    log.info("=" * 60)
    
    tasks = ["reacher_easy", "cartpole_swingup"]
    
    conditions = [
        # (name, n_ensemble, alpha, beta)
        ("FULL",       5, 5.0, 2.0),   # baseline
        ("SINGLE",     1, 5.0, 0.0),   # single model, no uncertainty
        ("NO_UNCERT",  5, 5.0, 0.0),   # ensemble but no penalty
        ("NO_REWARD",  5, 0.0, 2.0),   # no reward prediction
    ]
    
    all_results = []
    t0 = time.time()
    
    for cond_name, n_ens, alpha, beta in conditions:
        log.info(f"\n{'='*60}")
        log.info(f"CONDITION: {cond_name} (ensemble={n_ens}, α={alpha}, β={beta})")
        log.info(f"{'='*60}")
        
        for task in tasks:
            log.info(f"\n  Task: {task}")
            results = evaluate_condition(
                task, cond_name, n_ens, alpha, beta, n_demos=5
            )
            all_results.extend(results)
    
    elapsed = time.time() - t0
    
    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"ABLATION RESULTS SUMMARY (elapsed: {elapsed/60:.1f} min)")
    log.info(f"{'='*70}")
    log.info(f"{'Condition':12s} {'Task':18s} {'Mode':8s} {'Avg Replay':>10s} {'Std':>8s} {'N':>3s}")
    log.info("-" * 65)
    
    summary = {}
    for cond_name, _, _, _ in conditions:
        for task in tasks:
            for mode in ["faithful", "shifted"]:
                key = (cond_name, task, mode)
                vals = [r["replay_return"] for r in all_results 
                        if r["condition"] == cond_name 
                        and r["task"] == task 
                        and r["mode"] == mode]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                n = len(vals)
                summary[key] = {"mean": mean_val, "std": std_val, "n": n}
                log.info(f"{cond_name:12s} {task:18s} {mode:8s} "
                         f"{mean_val:10.1f} {std_val:8.1f} {n:3d}")
    
    # Save
    out_path = f"{RESULTS_DIR}/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": all_results,
            "summary": {str(k): v for k, v in summary.items()},
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
