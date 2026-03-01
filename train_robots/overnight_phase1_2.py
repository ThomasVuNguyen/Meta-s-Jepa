"""
Overnight Script — Phase 1 + 2: Multi-Task Data Collection & Ensemble Training
================================================================================
Runs on Prime Intellect A100. Estimated runtime: 6-7 hours.

What this does:
  Phase 1: Collect 5K episodes each for reacher-easy, point_mass-easy, cartpole-swingup
           Then encode all frames through V-JEPA 2 → embeddings
  Phase 2: Train 5× ensemble dynamics models per task
           Train reward models per task

Output: /root/vjepa_mvp/data/  → datasets (.npz)
        /root/vjepa_mvp/models/ → dynamics ensemble + reward models (.pt)
        /root/vjepa_mvp/logs/   → training logs

Usage: MUJOCO_GL=osmesa HF_TOKEN=xxx python3 overnight_phase1_2.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────

TASKS = {
    "reacher_easy": {
        "domain": "reacher",
        "task": "easy",
        "action_dim": 2,
        "expert_type": "p_controller",
    },
    "point_mass_easy": {
        "domain": "point_mass",
        "task": "easy",
        "action_dim": 2,
        "expert_type": "p_controller",
    },
    "cartpole_swingup": {
        "domain": "cartpole",
        "task": "swingup",
        "action_dim": 1,
        "expert_type": "energy",
    },
}

EPISODES_PER_TASK = 5000
MAX_STEPS = 200
BATCH_SIZE = 64  # V-JEPA encoding batch size (A100 can handle more)
EPSILON = 0.3  # exploration noise for data collection
LATENT_DIM = 1024
HIDDEN_DIM = 512
ENSEMBLE_SIZE = 5
DYNAMICS_EPOCHS = 100
REWARD_EPOCHS = 50

BASE_DIR = Path("/root/vjepa_mvp")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Expert Policies ──────────────────────────────────────────────────

def reacher_expert(obs, action_spec, eps):
    """P-controller for reacher: move toward target."""
    if np.random.rand() < eps:
        return np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum
        ).astype(np.float32)
    return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)


def point_mass_expert(obs, action_spec, eps):
    """P-controller for point_mass: move toward target."""
    if np.random.rand() < eps:
        return np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum
        ).astype(np.float32)
    # point_mass obs has 'position' and 'velocity'
    # target is at origin (0,0) by default
    pos = obs.get("position", np.zeros(2))
    vel = obs.get("velocity", np.zeros(2))
    # PD controller toward origin
    action = -2.0 * pos - 0.5 * vel
    return np.clip(action, action_spec.minimum, action_spec.maximum).astype(np.float32)


def cartpole_expert(obs, action_spec, eps):
    """Energy-based controller for cartpole swingup."""
    if np.random.rand() < eps:
        return np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum
        ).astype(np.float32)
    # obs keys: 'position' (cart_pos, cos, sin), 'velocity' (cart_vel, pole_vel)
    position = obs.get("position", np.zeros(3))
    velocity = obs.get("velocity", np.zeros(2))
    
    if len(position) >= 3:
        cos_angle = position[1] if len(position) > 1 else 0
        sin_angle = position[2] if len(position) > 2 else 0
        cart_pos = position[0]
    else:
        cos_angle, sin_angle, cart_pos = 1.0, 0.0, 0.0
    
    cart_vel = velocity[0] if len(velocity) > 0 else 0
    pole_vel = velocity[1] if len(velocity) > 1 else 0
    
    # Energy-based swing-up + balance
    angle = np.arctan2(sin_angle, cos_angle)
    energy = 0.5 * pole_vel**2 + 9.81 * (cos_angle - 1)
    
    if abs(angle) < 0.3:
        # Near top: balance with PD
        action = 5.0 * angle + 1.0 * pole_vel - 2.0 * cart_pos - 1.0 * cart_vel
    else:
        # Swing up: pump energy
        action = 3.0 * pole_vel * cos_angle + 0.5 * energy * np.sign(pole_vel)
    
    return np.clip(np.array([action]), action_spec.minimum, action_spec.maximum).astype(np.float32)


EXPERT_POLICIES = {
    "p_controller": reacher_expert,
    "energy": cartpole_expert,
}


def get_expert(task_config):
    """Return the appropriate expert policy for a task."""
    expert_type = task_config["expert_type"]
    if expert_type == "p_controller" and task_config["domain"] == "point_mass":
        return point_mass_expert
    return EXPERT_POLICIES[expert_type]


# ── V-JEPA 2 Encoder ────────────────────────────────────────────────

def load_vjepa(device="cuda"):
    """Load frozen V-JEPA 2 ViT-L encoder."""
    from transformers import AutoModel
    log("Loading V-JEPA 2 ViT-L...")
    
    cache_dir = "/root/vjepa_mvp/cache/hf"
    os.makedirs(cache_dir, exist_ok=True)
    
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device, dtype=torch.float16).eval()
    
    log(f"V-JEPA 2 loaded on {device}")
    return model


def encode_episode_frames(vjepa, frames, device="cuda", batch_size=BATCH_SIZE):
    """Encode all frames from one episode using sliding windows."""
    from torchvision import transforms
    from PIL import Image
    
    ET = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    frame_tensors = [ET(Image.fromarray(f)) for f in frames]
    n_frames = len(frames)
    embeddings = []
    
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        clips = []
        for t in range(start_idx, end_idx):
            window_start = max(0, t - 7)
            window = frame_tensors[window_start:t + 1]
            while len(window) < 8:
                window = [window[0]] + window
            clips.append(torch.stack(window))
        
        clips_t = torch.stack(clips).to(device, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=clips_t)
            embs = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
        embeddings.append(embs)
    
    return np.concatenate(embeddings, axis=0)


# ── Phase 1: Data Collection ────────────────────────────────────────

def collect_task_data(task_name, task_config, vjepa, device="cuda"):
    """Collect episodes and encode with V-JEPA 2 for one task."""
    from dm_control import suite
    
    domain = task_config["domain"]
    task = task_config["task"]
    expert = get_expert(task_config)
    
    log(f"=== Collecting {EPISODES_PER_TASK} episodes for {task_name} ===")
    
    all_z_t, all_a_t, all_z_next, all_rewards = [], [], [], []
    total_reward_sum = 0
    successes = 0
    
    start_time = time.time()
    
    for ep in range(EPISODES_PER_TASK):
        env = suite.load(domain, task, task_kwargs={"random": ep})
        action_spec = env.action_spec()
        
        time_step = env.reset()
        obs = time_step.observation
        
        frames = [env.physics.render(height=224, width=224, camera_id=0).copy()]
        actions = []
        rewards = []
        
        for step in range(MAX_STEPS):
            action = expert(obs, action_spec, EPSILON)
            actions.append(action.copy())
            time_step = env.step(action)
            obs = time_step.observation
            rewards.append(float(time_step.reward or 0.0))
            frames.append(env.physics.render(height=224, width=224, camera_id=0).copy())
        
        # Encode frames
        ep_embs = encode_episode_frames(vjepa, frames, device)
        z_t = ep_embs[:-1]
        z_next = ep_embs[1:]
        a_t = np.array(actions, dtype=np.float32)
        r_t = np.array(rewards, dtype=np.float32)
        
        all_z_t.append(z_t)
        all_a_t.append(a_t)
        all_z_next.append(z_next)
        all_rewards.append(r_t)
        
        ep_reward = sum(rewards)
        total_reward_sum += ep_reward
        if ep_reward > 100:
            successes += 1
        
        if (ep + 1) % 100 == 0 or ep == 0:
            elapsed = time.time() - start_time
            eps_per_min = (ep + 1) / (elapsed / 60)
            remaining = (EPISODES_PER_TASK - ep - 1) / eps_per_min
            avg_r = total_reward_sum / (ep + 1)
            sr = successes / (ep + 1)
            log(f"  [{task_name}] Ep {ep+1:4d}/{EPISODES_PER_TASK} | "
                f"Avg R: {avg_r:.1f} | SR: {sr:.1%} | "
                f"{eps_per_min:.1f} ep/min | ETA: {remaining:.0f} min")
        
        # Save checkpoint every 1000 episodes
        if (ep + 1) % 1000 == 0:
            save_dataset(task_name, all_z_t, all_a_t, all_z_next, all_rewards,
                        ep + 1, partial=True)
    
    # Final save
    elapsed = time.time() - start_time
    log(f"  [{task_name}] Complete: {EPISODES_PER_TASK} episodes in {elapsed/60:.1f} min")
    save_dataset(task_name, all_z_t, all_a_t, all_z_next, all_rewards, EPISODES_PER_TASK)


def save_dataset(task_name, z_t_list, a_t_list, z_next_list, r_list, n_eps, partial=False):
    """Save dataset to disk."""
    z_t = np.concatenate(z_t_list, axis=0)
    a_t = np.concatenate(a_t_list, axis=0)
    z_next = np.concatenate(z_next_list, axis=0)
    rewards = np.concatenate(r_list, axis=0)
    
    suffix = f"_partial_{n_eps}" if partial else ""
    out_path = DATA_DIR / f"{task_name}{suffix}.npz"
    np.savez_compressed(str(out_path), z_t=z_t, a_t=a_t, z_next=z_next, rewards=rewards)
    
    log(f"  Saved {task_name}: {len(z_t):,} transitions → {out_path.name} "
        f"({out_path.stat().st_size / 1e9:.2f} GB)")


# ── Model Definitions ────────────────────────────────────────────────

class DynamicsModel(nn.Module):
    """MLP dynamics: z_t+1 = z_t + f(z_t, a_t). ~1.2M params."""
    def __init__(self, latent_dim=LATENT_DIM, action_dim=2, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z_t, a_t):
        x = torch.cat([z_t, a_t], dim=-1)
        return z_t + self.net(x)


class RewardModel(nn.Module):
    """Reward predictor: R(z_t, a_t) → scalar reward."""
    def __init__(self, latent_dim=LATENT_DIM, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z_t, a_t):
        x = torch.cat([z_t, a_t], dim=-1)
        return self.net(x).squeeze(-1)


# ── Phase 2: Ensemble Training ──────────────────────────────────────

def train_dynamics_ensemble(task_name, action_dim, device="cuda"):
    """Train 5 independent dynamics models on task data."""
    log(f"=== Training dynamics ensemble for {task_name} ===")
    
    # Load data
    data_path = DATA_DIR / f"{task_name}.npz"
    data = np.load(str(data_path))
    z_t = torch.tensor(data["z_t"], dtype=torch.float32)
    a_t = torch.tensor(data["a_t"], dtype=torch.float32)
    z_next = torch.tensor(data["z_next"], dtype=torch.float32)
    
    n = len(z_t)
    split = int(n * 0.95)
    
    log(f"  Data: {n:,} transitions, action_dim={action_dim}")
    
    task_model_dir = MODEL_DIR / task_name
    task_model_dir.mkdir(parents=True, exist_ok=True)
    
    for ens_idx in range(ENSEMBLE_SIZE):
        log(f"  Training ensemble member {ens_idx+1}/{ENSEMBLE_SIZE}...")
        
        # Different random seed for each ensemble member
        torch.manual_seed(42 + ens_idx * 1000)
        np.random.seed(42 + ens_idx * 1000)
        
        # Shuffle data differently for each member
        perm = torch.randperm(n)
        z_t_s = z_t[perm]
        a_t_s = a_t[perm]
        z_next_s = z_next[perm]
        
        train_z = z_t_s[:split].to(device)
        train_a = a_t_s[:split].to(device)
        train_zn = z_next_s[:split].to(device)
        val_z = z_t_s[split:].to(device)
        val_a = a_t_s[split:].to(device)
        val_zn = z_next_s[split:].to(device)
        
        model = DynamicsModel(action_dim=action_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, DYNAMICS_EPOCHS)
        
        batch_size = 4096
        best_val_loss = float("inf")
        
        for epoch in range(DYNAMICS_EPOCHS):
            model.train()
            total_loss = 0
            n_batches = 0
            
            perm_epoch = torch.randperm(len(train_z), device=device)
            for i in range(0, len(train_z), batch_size):
                idx = perm_epoch[i:i + batch_size]
                pred = model(train_z[idx], train_a[idx])
                loss = nn.functional.mse_loss(pred, train_zn[idx])
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_train_loss = total_loss / n_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(val_z, val_a)
                val_loss = nn.functional.mse_loss(val_pred, val_zn).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                          str(task_model_dir / f"dynamics_ens_{ens_idx}.pt"))
            
            if (epoch + 1) % 20 == 0:
                log(f"    Ens {ens_idx} Epoch {epoch+1}/{DYNAMICS_EPOCHS} | "
                    f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Best: {best_val_loss:.6f}")
        
        log(f"  Ensemble member {ens_idx} done. Best val loss: {best_val_loss:.6f}")
    
    log(f"  Ensemble training complete for {task_name}")


def train_reward_model(task_name, action_dim, device="cuda"):
    """Train reward predictor on task data."""
    log(f"=== Training reward model for {task_name} ===")
    
    data_path = DATA_DIR / f"{task_name}.npz"
    data = np.load(str(data_path))
    z_t = torch.tensor(data["z_t"], dtype=torch.float32)
    a_t = torch.tensor(data["a_t"], dtype=torch.float32)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32)
    
    n = len(z_t)
    split = int(n * 0.95)
    
    perm = torch.randperm(n)
    train_z = z_t[perm[:split]].to(device)
    train_a = a_t[perm[:split]].to(device)
    train_r = rewards[perm[:split]].to(device)
    val_z = z_t[perm[split:]].to(device)
    val_a = a_t[perm[split:]].to(device)
    val_r = rewards[perm[split:]].to(device)
    
    model = RewardModel(action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    batch_size = 4096
    best_val_loss = float("inf")
    
    task_model_dir = MODEL_DIR / task_name
    task_model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(REWARD_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        
        perm_epoch = torch.randperm(len(train_z), device=device)
        for i in range(0, len(train_z), batch_size):
            idx = perm_epoch[i:i + batch_size]
            pred = model(train_z[idx], train_a[idx])
            loss = nn.functional.mse_loss(pred, train_r[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_z, val_a)
            val_loss = nn.functional.mse_loss(val_pred, val_r).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      str(task_model_dir / "reward_model.pt"))
        
        if (epoch + 1) % 10 == 0:
            log(f"  Epoch {epoch+1}/{REWARD_EPOCHS} | Train: {avg_loss:.6f} | "
                f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
    
    # Analyze reward distribution
    model.eval()
    with torch.no_grad():
        val_pred = model(val_z, val_a)
    
    r_mean = val_r.mean().item()
    r_std = val_r.std().item()
    nonzero_pct = (val_r > 0).float().mean().item()
    
    log(f"  Reward stats: mean={r_mean:.4f}, std={r_std:.4f}, "
        f"nonzero={nonzero_pct:.1%}")
    log(f"  Reward model done. Best val MSE: {best_val_loss:.6f}")
    
    return {"val_mse": best_val_loss, "r_mean": r_mean, "r_std": r_std,
            "nonzero_pct": nonzero_pct}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("OVERNIGHT SCRIPT: Phase 1 (Data Collection) + Phase 2 (Ensemble Training)")
    log("=" * 70)
    
    # Setup
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    overall_start = time.time()
    results = {}
    
    # ── Phase 1: Data Collection ─────────────────────────────────
    log("\n" + "=" * 70)
    log("PHASE 1: Multi-Task Data Collection + V-JEPA 2 Encoding")
    log("=" * 70)
    
    vjepa = load_vjepa(device)
    
    for task_name, task_config in TASKS.items():
        task_start = time.time()
        
        # Check if data already exists (resume support)
        data_path = DATA_DIR / f"{task_name}.npz"
        if data_path.exists():
            data = np.load(str(data_path))
            n_transitions = len(data["z_t"])
            expected = EPISODES_PER_TASK * MAX_STEPS
            if n_transitions >= expected * 0.95:  # allow 5% tolerance
                log(f"  [{task_name}] Data already exists ({n_transitions:,} transitions). Skipping.")
                continue
            else:
                log(f"  [{task_name}] Partial data found ({n_transitions:,}/{expected:,}). Re-collecting.")
        
        collect_task_data(task_name, task_config, vjepa, device)
        
        task_elapsed = time.time() - task_start
        log(f"  [{task_name}] Data collection took {task_elapsed/60:.1f} min")
    
    # Free V-JEPA from GPU memory
    del vjepa
    torch.cuda.empty_cache()
    log("V-JEPA 2 unloaded from GPU")
    
    # ── Phase 2: Ensemble Training ───────────────────────────────
    log("\n" + "=" * 70)
    log("PHASE 2: Ensemble Dynamics + Reward Model Training")
    log("=" * 70)
    
    for task_name, task_config in TASKS.items():
        action_dim = task_config["action_dim"]
        
        # Check if ensemble already trained
        task_model_dir = MODEL_DIR / task_name
        existing_models = list(task_model_dir.glob("dynamics_ens_*.pt")) if task_model_dir.exists() else []
        if len(existing_models) >= ENSEMBLE_SIZE:
            log(f"  [{task_name}] Ensemble already trained ({len(existing_models)} models). Skipping.")
        else:
            train_dynamics_ensemble(task_name, action_dim, device)
        
        # Reward model
        reward_path = task_model_dir / "reward_model.pt" if task_model_dir.exists() else None
        if reward_path and reward_path.exists():
            log(f"  [{task_name}] Reward model already trained. Skipping.")
        else:
            reward_stats = train_reward_model(task_name, action_dim, device)
            results[f"{task_name}_reward"] = reward_stats
    
    # ── Summary ──────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    
    log("\n" + "=" * 70)
    log("OVERNIGHT SCRIPT COMPLETE!")
    log("=" * 70)
    log(f"Total runtime: {total_elapsed/3600:.1f} hours")
    log(f"Estimated cost: ${total_elapsed/3600 * 0.90:.2f}")
    
    # Print what was created
    log("\nData files:")
    for f in sorted(DATA_DIR.glob("*.npz")):
        size_gb = f.stat().st_size / 1e9
        data = np.load(str(f))
        log(f"  {f.name}: {len(data['z_t']):,} transitions ({size_gb:.2f} GB)")
    
    log("\nModel files:")
    for task_dir in sorted(MODEL_DIR.iterdir()):
        if task_dir.is_dir():
            models = list(task_dir.glob("*.pt"))
            log(f"  {task_dir.name}/: {len(models)} models")
            for m in sorted(models):
                log(f"    {m.name} ({m.stat().st_size / 1e6:.1f} MB)")
    
    # Save summary
    summary = {
        "total_runtime_hours": total_elapsed / 3600,
        "estimated_cost_usd": total_elapsed / 3600 * 0.90,
        "tasks": list(TASKS.keys()),
        "episodes_per_task": EPISODES_PER_TASK,
        "ensemble_size": ENSEMBLE_SIZE,
        "results": results,
        "completed_at": datetime.now().isoformat(),
    }
    with open(str(LOG_DIR / "overnight_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    log(f"\nSummary saved to {LOG_DIR / 'overnight_summary.json'}")
    log("Ready for Phase 3 (agent training) when you wake up! 🚀")


if __name__ == "__main__":
    main()
