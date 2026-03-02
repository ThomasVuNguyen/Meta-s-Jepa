#!/usr/bin/env python3
"""
OVERNIGHT MULTI-TASK EVALUATION
================================
Fully autonomous: collects data, trains models, evaluates agent on new DMC tasks.
Expected runtime: 8-12 hours on A100.

Tasks (new):
  - walker_walk     (a_dim=6, complex → 5 ensemble)
  - cheetah_run     (a_dim=6, complex → 5 ensemble)
  - finger_spin     (a_dim=2, simple  → 1 model)
  - hopper_hop      (a_dim=4, medium  → 3 ensemble)
  - cup_catch       (a_dim=2, simple  → 1 model)

Also re-evaluates existing tasks with more seeds:
  - reacher_easy    (a_dim=2, existing models)
  - cartpole_swingup (a_dim=1, existing models)
  - point_mass_easy  (a_dim=2, existing models)
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import sys, time, json, logging, gc, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
BASE = "/home/ubuntu/vjepa_mvp"
MODEL_DIR = f"{BASE}/models"
DATA_DIR = f"{BASE}/data"
RESULTS_DIR = f"{BASE}/results"
for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────
LATENT_DIM = 1024
HIDDEN_DIM = 512

# Tasks: (name, n_episodes, n_ensemble)
# Based on ablation findings: ensemble helps complex tasks, single model fine for simple
NEW_TASKS = [
    ("walker_walk",       500, 5),   # complex, 6D action
    ("cheetah_run",       500, 5),   # complex, 6D action
    ("finger_spin",       500, 1),   # simple, 2D action
    ("hopper_hop",        500, 3),   # medium, 4D action
    ("cup_catch",         500, 1),   # simple, 2D action
]

EXISTING_TASKS = [
    ("reacher_easy",      0, 5),     # already have models
    ("cartpole_swingup",  0, 5),     # already have models
    ("point_mass_easy",   0, 5),     # already have models
]

# ── Model definitions (matching overnight_v3.py) ────────────────────
class DynamicsModel(nn.Module):
    def __init__(self, ldim=LATENT_DIM, adim=2, hdim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim+adim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, ldim))
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))

class RewardModel(nn.Module):
    def __init__(self, ldim=LATENT_DIM, adim=2, hdim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim+adim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, 1))
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
    log.info("Loading V-JEPA 2...")
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
            # V-JEPA expects (B, T, C, H, W)
            tensors = tensors.unsqueeze(1).to(self.device)
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

# ── DMC Environment ─────────────────────────────────────────────────
def make_env(task, seed=0):
    from dm_control import suite
    domain, task_name = task.split("_", 1)
    if domain == "point" and task_name.startswith("mass"):
        domain = "point_mass"
        task_name = task_name[5:]
    env = suite.load(domain, task_name, task_kwargs={"random": seed})
    return env

def render_frame(env, size=224):
    return env.physics.render(height=size, width=size, camera_id=0)

def close_env(env):
    try:
        if hasattr(env, '_physics') and env._physics is not None:
            env._physics.free()
    except Exception:
        pass

# ═════════════════════════════════════════════════════════════════════
# PHASE 1: Data Collection + V-JEPA Encoding
# ═════════════════════════════════════════════════════════════════════
def collect_and_encode(task, n_episodes, n_steps=200):
    """Collect random episodes, encode with V-JEPA, save as .npz"""
    data_path = f"{DATA_DIR}/{task}_{n_episodes}.npz"
    if os.path.exists(data_path):
        log.info(f"  Data exists: {data_path}, loading...")
        data = np.load(data_path)
        return data['z_curr'], data['actions'], data['rewards'], data['z_next']
    
    encoder, device = get_encoder()
    env = make_env(task, seed=0)
    a_spec = env.action_spec()
    a_dim = a_spec.shape[0]
    close_env(env)
    
    all_z_curr, all_actions, all_rewards, all_z_next = [], [], [], []
    
    t0 = time.time()
    for ep in range(n_episodes):
        env = make_env(task, seed=ep)
        ts = env.reset()
        
        frames = [render_frame(env)]
        actions = []
        rewards = []
        
        for step in range(n_steps):
            action = np.random.uniform(a_spec.minimum, a_spec.maximum, size=a_dim)
            ts = env.step(action)
            frames.append(render_frame(env))
            actions.append(action)
            rewards.append(ts.reward or 0.0)
        
        close_env(env)
        
        # Encode all frames
        embeddings = encoder.encode_frames(frames).cpu().numpy()  # (201, 1024)
        
        for i in range(n_steps):
            all_z_curr.append(embeddings[i])
            all_actions.append(actions[i])
            all_rewards.append(rewards[i])
            all_z_next.append(embeddings[i+1])
        
        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed * 60
            log.info(f"  {task}: {ep+1}/{n_episodes} episodes "
                     f"({rate:.1f} ep/min, {elapsed/60:.1f} min elapsed)")
    
    z_curr = np.array(all_z_curr, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    z_next = np.array(all_z_next, dtype=np.float32)
    
    np.savez_compressed(data_path, z_curr=z_curr, actions=actions,
                        rewards=rewards, z_next=z_next)
    
    total_time = time.time() - t0
    log.info(f"  {task}: Done! {n_episodes} episodes, {len(z_curr)} transitions, "
             f"{total_time/60:.1f} min")
    
    return z_curr, actions, rewards, z_next

# ═════════════════════════════════════════════════════════════════════
# PHASE 2: Train Dynamics + Reward Models
# ═════════════════════════════════════════════════════════════════════
def train_models(task, z_curr, actions, rewards, z_next, n_ensemble=5,
                 n_epochs=100, batch_size=512, lr=3e-4):
    """Train dynamics ensemble + reward model."""
    task_dir = f"{MODEL_DIR}/{task}"
    os.makedirs(task_dir, exist_ok=True)
    
    # Check if models already exist
    if os.path.exists(f"{task_dir}/dyn_0.pt") and os.path.exists(f"{task_dir}/reward.pt"):
        log.info(f"  Models exist for {task}, skipping training.")
        return
    
    device = torch.device("cuda")
    a_dim = actions.shape[1]
    
    # Convert to tensors
    z_t = torch.tensor(z_curr, device=device)
    a_t = torch.tensor(actions, device=device)
    r_t = torch.tensor(rewards, device=device)
    z_next_t = torch.tensor(z_next, device=device)
    
    # Train/val split (90/10)
    n = len(z_t)
    perm = torch.randperm(n)
    n_val = n // 10
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    # ── Train dynamics models ────────────────────────────────────────
    for ens_i in range(n_ensemble):
        model_path = f"{task_dir}/dyn_{ens_i}.pt"
        if os.path.exists(model_path):
            log.info(f"  dyn_{ens_i} exists, skipping.")
            continue
            
        model = DynamicsModel(LATENT_DIM, a_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        
        # Bootstrap: different random subset for each ensemble member
        boot_idx = train_idx[torch.randint(len(train_idx), (len(train_idx),))]
        
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            model.train()
            # Shuffle
            perm_train = boot_idx[torch.randperm(len(boot_idx))]
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(perm_train), batch_size):
                idx = perm_train[i:i+batch_size]
                pred = model(z_t[idx], a_t[idx])
                loss = (pred - z_next_t[idx]).pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # Validation
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(z_t[val_idx], a_t[val_idx])
                    val_loss = (val_pred - z_next_t[val_idx]).pow(2).mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                
                log.info(f"    dyn_{ens_i} epoch {epoch+1}: train={epoch_loss/n_batches:.6f}, "
                         f"val={val_loss:.6f}, best={best_val_loss:.6f}")
        
        # Save final if never saved
        if not os.path.exists(model_path):
            torch.save(model.state_dict(), model_path)
        
        log.info(f"  dyn_{ens_i} done: best val MSE = {best_val_loss:.6f}")
    
    # ── Train reward model ───────────────────────────────────────────
    reward_path = f"{task_dir}/reward.pt"
    if not os.path.exists(reward_path):
        model = RewardModel(LATENT_DIM, a_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            model.train()
            perm_train = train_idx[torch.randperm(len(train_idx))]
            
            for i in range(0, len(perm_train), batch_size):
                idx = perm_train[i:i+batch_size]
                pred = model(z_t[idx], a_t[idx])
                loss = (pred - r_t[idx]).pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(z_t[val_idx], a_t[val_idx])
                    val_loss = (val_pred - r_t[val_idx]).pow(2).mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), reward_path)
        
        if not os.path.exists(reward_path):
            torch.save(model.state_dict(), reward_path)
        
        log.info(f"  reward done: best val MSE = {best_val_loss:.6f}")
    
    # Free GPU memory
    del z_t, a_t, r_t, z_next_t
    torch.cuda.empty_cache()
    gc.collect()

# ═════════════════════════════════════════════════════════════════════
# PHASE 3: Teach-by-Showing Agent Evaluation
# ═════════════════════════════════════════════════════════════════════
def load_task_models(task, device, n_ensemble=5):
    task_dir = f"{MODEL_DIR}/{task}"
    ckpt = torch.load(f"{task_dir}/dyn_0.pt", map_location=device, weights_only=True)
    first_weight = list(ckpt.values())[0]
    a_dim = first_weight.shape[1] - LATENT_DIM
    
    # Load however many ensemble members exist
    dyn_models = []
    for i in range(n_ensemble):
        path = f"{task_dir}/dyn_{i}.pt"
        if not os.path.exists(path):
            break
        m = DynamicsModel(LATENT_DIM, a_dim).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        dyn_models.append(m)
    
    reward_model = RewardModel(LATENT_DIM, a_dim).to(device)
    reward_model.load_state_dict(torch.load(f"{task_dir}/reward.pt", map_location=device, weights_only=True))
    reward_model.eval()
    
    return dyn_models, reward_model, a_dim

@torch.no_grad()
def cem_plan(z_curr, goal_z_seq, dyn_models, reward_model,
             a_dim, a_min, a_max, device,
             n_candidates=500, n_elite=50, n_iters=5, horizon=8,
             alpha=5.0, beta=2.0):
    mu = torch.zeros(horizon, a_dim, device=device)
    std = torch.ones(horizon, a_dim, device=device) * 0.5
    a_min_t = torch.tensor(a_min, device=device, dtype=torch.float32)
    a_max_t = torch.tensor(a_max, device=device, dtype=torch.float32)
    
    for _ in range(n_iters):
        noise = torch.randn(n_candidates, horizon, a_dim, device=device)
        actions = mu.unsqueeze(0) + std.unsqueeze(0) * noise
        actions = actions.clamp(a_min_t, a_max_t)
        
        scores = torch.zeros(n_candidates, device=device)
        z = z_curr.unsqueeze(0).expand(n_candidates, -1)
        
        for t in range(horizon):
            a = actions[:, t]
            preds = torch.stack([m(z, a) for m in dyn_models])
            z_next = preds.mean(0)
            
            if beta > 0 and len(dyn_models) > 1:
                scores -= beta * preds.std(0).mean(-1)
            if alpha > 0:
                scores += alpha * reward_model(z, a)
            
            goal_idx = min(t, len(goal_z_seq) - 1)
            scores -= (z_next - goal_z_seq[goal_idx]).pow(2).sum(-1).sqrt()
            z = z_next
        
        elite_idx = scores.topk(n_elite).indices
        elite_actions = actions[elite_idx]
        mu = elite_actions.mean(0)
        std = elite_actions.std(0).clamp(min=0.05)
    
    return mu[0].cpu().numpy()

def evaluate_task(task, n_ensemble, n_demos=10, n_steps=200, lookahead=15):
    """Evaluate teach-by-showing agent on a task."""
    encoder, device = get_encoder()
    dyn_models, reward_model, a_dim = load_task_models(task, device, n_ensemble)
    
    log.info(f"  Loaded {len(dyn_models)}× dynamics + reward for {task} (a_dim={a_dim})")
    
    # Use FULL config from ablation (α=5.0, β=2.0)
    alpha, beta = 5.0, 2.0
    results = []
    
    for demo_idx in range(n_demos):
        seed = demo_idx
        
        # Record random demo
        env = make_env(task, seed=seed)
        ts = env.reset()
        frames = [render_frame(env)]
        demo_rewards = []
        
        for step in range(n_steps):
            action = np.random.uniform(env.action_spec().minimum, 
                                        env.action_spec().maximum,
                                        size=a_dim)
            ts = env.step(action)
            frames.append(render_frame(env))
            demo_rewards.append(ts.reward or 0.0)
        
        close_env(env)
        demo_return = sum(demo_rewards)
        
        # Encode demo
        goal_embeddings = encoder.encode_frames(frames)
        
        for mode, replay_seed in [("faithful", seed), ("shifted", seed + 1000)]:
            env = make_env(task, seed=replay_seed)
            ts = env.reset()
            z_curr = encoder.encode_frames([render_frame(env)])[0]
            
            a_spec = env.action_spec()
            total_reward = 0.0
            
            for step in range(n_steps):
                start_idx = min(step, len(goal_embeddings) - lookahead)
                end_idx = min(start_idx + lookahead, len(goal_embeddings))
                goal_seq = goal_embeddings[start_idx:end_idx]
                
                action = cem_plan(z_curr, goal_seq, dyn_models, reward_model,
                                  a_dim, a_spec.minimum, a_spec.maximum, device,
                                  alpha=alpha, beta=beta)
                
                ts = env.step(action)
                total_reward += (ts.reward or 0.0)
                z_curr = encoder.encode_frames([render_frame(env)])[0]
            
            close_env(env)
            
            results.append({
                "task": task, "mode": mode,
                "demo_seed": seed, "replay_seed": replay_seed,
                "demo_return": round(demo_return, 1),
                "replay_return": round(total_reward, 1),
                "n_ensemble": len(dyn_models),
                "alpha": alpha, "beta": beta,
            })
            
            log.info(f"    {mode:8s} seed={seed}: demo={demo_return:.1f} → replay={total_reward:.1f}")
    
    return results

# ═════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════
def main():
    t0_total = time.time()
    log.info("=" * 70)
    log.info("OVERNIGHT MULTI-TASK EVALUATION")
    log.info("=" * 70)
    
    all_results = []
    task_summaries = {}
    
    # ── Process new tasks: collect + train + evaluate ────────────────
    for task, n_episodes, n_ensemble in NEW_TASKS:
        try:
            log.info(f"\n{'='*70}")
            log.info(f"TASK: {task} ({n_episodes} episodes, {n_ensemble} ensemble)")
            log.info(f"{'='*70}")
            
            # Phase 1: Collect and encode
            t0 = time.time()
            log.info(f"\n  [1/3] Collecting {n_episodes} episodes...")
            z_curr, actions, rewards, z_next = collect_and_encode(task, n_episodes)
            collect_time = time.time() - t0
            log.info(f"  Collection done: {len(z_curr)} transitions, {collect_time/60:.1f} min")
            
            # Phase 2: Train models
            t0 = time.time()
            log.info(f"\n  [2/3] Training {n_ensemble}× dynamics + reward...")
            train_models(task, z_curr, actions, rewards, z_next, n_ensemble=n_ensemble)
            train_time = time.time() - t0
            log.info(f"  Training done: {train_time/60:.1f} min")
            
            # Free data memory
            del z_curr, actions, rewards, z_next
            gc.collect()
            
            # Phase 3: Evaluate agent
            t0 = time.time()
            log.info(f"\n  [3/3] Evaluating teach-by-showing agent (10 demos)...")
            results = evaluate_task(task, n_ensemble, n_demos=10)
            eval_time = time.time() - t0
            log.info(f"  Evaluation done: {eval_time/60:.1f} min")
            
            all_results.extend(results)
            
            # Summary for this task
            faithful = [r["replay_return"] for r in results if r["mode"] == "faithful"]
            shifted = [r["replay_return"] for r in results if r["mode"] == "shifted"]
            task_summaries[task] = {
                "n_episodes": n_episodes,
                "n_ensemble": n_ensemble,
                "faithful_mean": round(np.mean(faithful), 1),
                "faithful_std": round(np.std(faithful), 1),
                "shifted_mean": round(np.mean(shifted), 1),
                "shifted_std": round(np.std(shifted), 1),
                "collect_min": round(collect_time/60, 1),
                "train_min": round(train_time/60, 1),
                "eval_min": round(eval_time/60, 1),
            }
            
        except Exception as e:
            log.error(f"  FAILED on {task}: {e}")
            traceback.print_exc()
            task_summaries[task] = {"error": str(e)}
    
    # ── Re-evaluate existing tasks with 10 seeds ─────────────────────
    for task, _, n_ensemble in EXISTING_TASKS:
        try:
            log.info(f"\n{'='*70}")
            log.info(f"RE-EVALUATING: {task} (existing models, {n_ensemble} ensemble)")
            log.info(f"{'='*70}")
            
            t0 = time.time()
            results = evaluate_task(task, n_ensemble, n_demos=10)
            eval_time = time.time() - t0
            
            all_results.extend(results)
            
            faithful = [r["replay_return"] for r in results if r["mode"] == "faithful"]
            shifted = [r["replay_return"] for r in results if r["mode"] == "shifted"]
            task_summaries[task] = {
                "n_ensemble": n_ensemble,
                "faithful_mean": round(np.mean(faithful), 1),
                "faithful_std": round(np.std(faithful), 1),
                "shifted_mean": round(np.mean(shifted), 1),
                "shifted_std": round(np.std(shifted), 1),
                "eval_min": round(eval_time/60, 1),
            }
            
        except Exception as e:
            log.error(f"  FAILED on {task}: {e}")
            traceback.print_exc()
            task_summaries[task] = {"error": str(e)}
    
    # ── Final Summary ────────────────────────────────────────────────
    total_time = time.time() - t0_total
    
    log.info(f"\n{'='*70}")
    log.info(f"OVERNIGHT EVALUATION COMPLETE")
    log.info(f"Total time: {total_time/3600:.1f} hours")
    log.info(f"{'='*70}")
    log.info(f"{'Task':20s} {'Ens':>3s} {'Faith Mean':>10s} {'±std':>6s} {'Shift Mean':>10s} {'±std':>6s}")
    log.info("-" * 60)
    
    for task in [t[0] for t in NEW_TASKS] + [t[0] for t in EXISTING_TASKS]:
        s = task_summaries.get(task, {})
        if "error" in s:
            log.info(f"{task:20s}  ERROR: {s['error'][:40]}")
        else:
            log.info(f"{task:20s} {s.get('n_ensemble','?'):>3} "
                     f"{s.get('faithful_mean',0):10.1f} {s.get('faithful_std',0):6.1f} "
                     f"{s.get('shifted_mean',0):10.1f} {s.get('shifted_std',0):6.1f}")
    
    # Save everything
    out = {
        "results": all_results,
        "summaries": task_summaries,
        "total_hours": round(total_time / 3600, 2),
    }
    
    out_path = f"{RESULTS_DIR}/overnight_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    log.info(f"\nResults saved to {out_path}")
    log.info("DONE.")

if __name__ == "__main__":
    main()
