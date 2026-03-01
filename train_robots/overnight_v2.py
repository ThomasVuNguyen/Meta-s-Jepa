"""
Overnight Script v2 — Optimized for single-GPU
===============================================
Key optimization: collect raw data FIRST (fast CPU loop), then batch encode on GPU.
This avoids the V-JEPA bottleneck during data collection.

Stage 1: Collect raw (frames, actions, rewards) for all tasks — ~30 min
Stage 2: Batch encode frames through V-JEPA 2 — ~3 hrs  
Stage 3: Train 5× ensemble dynamics per task — ~2 hrs
Stage 4: Train reward models per task — ~15 min

Total: ~6 hrs on A100

Usage: MUJOCO_GL=osmesa HF_TOKEN=xxx python3 overnight_v2.py
"""

import os
import sys
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────

TASKS = {
    "reacher_easy": {
        "domain": "reacher", "task": "easy",
        "action_dim": 2, "expert_type": "reacher",
    },
    "point_mass_easy": {
        "domain": "point_mass", "task": "easy",
        "action_dim": 2, "expert_type": "point_mass",
    },
    "cartpole_swingup": {
        "domain": "cartpole", "task": "swingup",
        "action_dim": 1, "expert_type": "cartpole",
    },
}

EPISODES_PER_TASK = 5000
MAX_STEPS = 200
ENCODE_BATCH = 128  # How many 8-frame clips to encode at once on A100
EPSILON = 0.3
LATENT_DIM = 1024
HIDDEN_DIM = 512
ENSEMBLE_SIZE = 5
DYNAMICS_EPOCHS = 100
REWARD_EPOCHS = 50

BASE_DIR = Path("/root/vjepa_mvp")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "raw"  # Raw frames before encoding
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Expert Policies ──────────────────────────────────────────────────

def reacher_expert(obs, action_spec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(action_spec.minimum, action_spec.maximum).astype(np.float32)
    return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

def point_mass_expert(obs, action_spec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(action_spec.minimum, action_spec.maximum).astype(np.float32)
    pos = obs.get("position", np.zeros(2))
    vel = obs.get("velocity", np.zeros(2))
    action = -2.0 * pos - 0.5 * vel
    return np.clip(action, action_spec.minimum, action_spec.maximum).astype(np.float32)

def cartpole_expert(obs, action_spec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(action_spec.minimum, action_spec.maximum).astype(np.float32)
    position = obs.get("position", np.zeros(3))
    velocity = obs.get("velocity", np.zeros(2))
    cos_a = position[1] if len(position) > 1 else 1.0
    sin_a = position[2] if len(position) > 2 else 0.0
    cart_pos = position[0]
    cart_vel = velocity[0] if len(velocity) > 0 else 0
    pole_vel = velocity[1] if len(velocity) > 1 else 0
    angle = np.arctan2(sin_a, cos_a)
    energy = 0.5 * pole_vel**2 + 9.81 * (cos_a - 1)
    if abs(angle) < 0.3:
        action = 5.0 * angle + 1.0 * pole_vel - 2.0 * cart_pos - 1.0 * cart_vel
    else:
        action = 3.0 * pole_vel * cos_a + 0.5 * energy * np.sign(pole_vel)
    return np.clip(np.array([action]), action_spec.minimum, action_spec.maximum).astype(np.float32)

EXPERTS = {"reacher": reacher_expert, "point_mass": point_mass_expert, "cartpole": cartpole_expert}


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1: Collect raw data (no GPU needed, very fast)
# ══════════════════════════════════════════════════════════════════════

def collect_raw_data(task_name, task_config):
    """Collect episodes, save raw actions + rewards + low-res frame indices.
    Frames saved as compressed numpy arrays (224x224 uint8).
    """
    from dm_control import suite
    
    domain = task_config["domain"]
    task = task_config["task"]
    expert = EXPERTS[task_config["expert_type"]]
    
    raw_task_dir = RAW_DIR / task_name
    raw_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Check resume
    done_marker = raw_task_dir / "collection_done.txt"
    if done_marker.exists():
        log(f"  [{task_name}] Raw data already collected. Skipping.")
        return
    
    log(f"  [{task_name}] Collecting {EPISODES_PER_TASK} episodes (raw)...")
    start = time.time()
    
    all_actions = []
    all_rewards = []
    total_reward = 0
    successes = 0
    
    # We'll save frames in chunks to avoid OOM
    CHUNK_SIZE = 500  # episodes per chunk
    chunk_frames = []
    chunk_idx = 0
    
    for ep in range(EPISODES_PER_TASK):
        env = suite.load(domain, task, task_kwargs={"random": ep})
        action_spec = env.action_spec()
        time_step = env.reset()
        obs = time_step.observation
        
        ep_frames = [env.physics.render(height=224, width=224, camera_id=0).copy()]
        ep_actions = []
        ep_rewards = []
        
        for step in range(MAX_STEPS):
            action = expert(obs, action_spec)
            ep_actions.append(action.copy())
            time_step = env.step(action)
            obs = time_step.observation
            ep_rewards.append(float(time_step.reward or 0.0))
            ep_frames.append(env.physics.render(height=224, width=224, camera_id=0).copy())
        
        # Store frames as numpy array: (201, 224, 224, 3) uint8
        chunk_frames.append(np.array(ep_frames, dtype=np.uint8))
        all_actions.append(np.array(ep_actions, dtype=np.float32))
        all_rewards.append(np.array(ep_rewards, dtype=np.float32))
        
        ep_reward = sum(ep_rewards)
        total_reward += ep_reward
        if ep_reward > 100:
            successes += 1
        
        # Save chunk when full
        if (ep + 1) % CHUNK_SIZE == 0:
            frames_arr = np.array(chunk_frames)  # (CHUNK, 201, 224, 224, 3)
            np.save(str(raw_task_dir / f"frames_chunk_{chunk_idx:03d}.npy"), frames_arr)
            chunk_frames = []
            chunk_idx += 1
            gc.collect()
        
        if (ep + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (ep + 1) / (elapsed / 60)
            eta = (EPISODES_PER_TASK - ep - 1) / rate
            sr = successes / (ep + 1)
            log(f"    Ep {ep+1:4d}/{EPISODES_PER_TASK} | R={total_reward/(ep+1):.1f} | "
                f"SR={sr:.1%} | {rate:.0f} ep/min | ETA {eta:.0f} min")
    
    # Save remaining frames
    if chunk_frames:
        frames_arr = np.array(chunk_frames)
        np.save(str(raw_task_dir / f"frames_chunk_{chunk_idx:03d}.npy"), frames_arr)
    
    # Save actions and rewards
    np.save(str(raw_task_dir / "actions.npy"), np.array(all_actions))  # (5000, 200, action_dim)
    np.save(str(raw_task_dir / "rewards.npy"), np.array(all_rewards))  # (5000, 200)
    
    done_marker.write_text(f"Done: {EPISODES_PER_TASK} episodes\n")
    
    elapsed = time.time() - start
    log(f"  [{task_name}] Raw collection done: {elapsed/60:.1f} min, "
        f"SR={successes/EPISODES_PER_TASK:.1%}, Avg R={total_reward/EPISODES_PER_TASK:.1f}")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2: Batch encode with V-JEPA 2 (GPU-intensive)
# ══════════════════════════════════════════════════════════════════════

def batch_encode_task(task_name, vjepa, device="cuda"):
    """Load raw frames chunk by chunk, encode with V-JEPA 2, save embeddings."""
    from torchvision import transforms
    from PIL import Image
    
    # Check if already done
    data_path = DATA_DIR / f"{task_name}.npz"
    if data_path.exists():
        data = np.load(str(data_path))
        if len(data["z_t"]) >= EPISODES_PER_TASK * MAX_STEPS * 0.95:
            log(f"  [{task_name}] Already encoded. Skipping.")
            return
    
    raw_task_dir = RAW_DIR / task_name
    
    ET = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    log(f"  [{task_name}] Encoding frames with V-JEPA 2...")
    start = time.time()
    
    # Load actions/rewards
    actions = np.load(str(raw_task_dir / "actions.npy"))  # (5000, 200, action_dim)
    rewards = np.load(str(raw_task_dir / "rewards.npy"))  # (5000, 200)
    
    all_z_t = []
    all_a_t = []
    all_z_next = []
    all_rewards = []
    
    # Process frame chunks
    chunk_files = sorted(raw_task_dir.glob("frames_chunk_*.npy"))
    ep_offset = 0
    
    for chunk_file in chunk_files:
        log(f"    Loading chunk {chunk_file.name}...")
        frames_chunk = np.load(str(chunk_file))  # (N_eps, 201, 224, 224, 3)
        n_eps = len(frames_chunk)
        
        for local_ep in range(n_eps):
            global_ep = ep_offset + local_ep
            ep_frames = frames_chunk[local_ep]  # (201, 224, 224, 3)
            
            # Convert frames to tensors
            frame_tensors = [ET(Image.fromarray(f)) for f in ep_frames]
            
            # Build 8-frame windows and encode in batches
            n_frames = len(frame_tensors)
            embeddings = []
            
            for batch_start in range(0, n_frames, ENCODE_BATCH):
                batch_end = min(batch_start + ENCODE_BATCH, n_frames)
                clips = []
                for t in range(batch_start, batch_end):
                    ws = max(0, t - 7)
                    window = frame_tensors[ws:t + 1]
                    while len(window) < 8:
                        window = [window[0]] + window
                    clips.append(torch.stack(window))
                
                clips_t = torch.stack(clips).to(device, dtype=torch.float16)
                with torch.no_grad():
                    out = vjepa(pixel_values_videos=clips_t)
                    embs = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
                embeddings.append(embs)
                del clips_t
            
            ep_embs = np.concatenate(embeddings, axis=0)  # (201, 1024)
            
            all_z_t.append(ep_embs[:-1])
            all_z_next.append(ep_embs[1:])
            all_a_t.append(actions[global_ep])
            all_rewards.append(rewards[global_ep])
            
            if (global_ep + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (global_ep + 1) / (elapsed / 60)
                eta = (EPISODES_PER_TASK - global_ep - 1) / rate
                log(f"    Encoded {global_ep+1}/{EPISODES_PER_TASK} | "
                    f"{rate:.1f} ep/min | ETA {eta:.0f} min")
        
        ep_offset += n_eps
        del frames_chunk
        gc.collect()
    
    # Save encoded dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    z_t = np.concatenate(all_z_t, axis=0)
    a_t = np.concatenate(all_a_t, axis=0)
    z_next = np.concatenate(all_z_next, axis=0)
    r = np.concatenate(all_rewards, axis=0)
    
    np.savez_compressed(str(data_path), z_t=z_t, a_t=a_t, z_next=z_next, rewards=r)
    
    elapsed = time.time() - start
    log(f"  [{task_name}] Encoding done: {len(z_t):,} transitions in {elapsed/60:.1f} min "
        f"({data_path.stat().st_size/1e9:.2f} GB)")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 & 4: Train ensemble dynamics + reward models
# ══════════════════════════════════════════════════════════════════════

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=2, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, z_t, a_t):
        return z_t + self.net(torch.cat([z_t, a_t], dim=-1))

class RewardModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, z_t, a_t):
        return self.net(torch.cat([z_t, a_t], dim=-1)).squeeze(-1)


def train_dynamics_ensemble(task_name, action_dim, device="cuda"):
    task_model_dir = MODEL_DIR / task_name
    task_model_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(task_model_dir.glob("dynamics_ens_*.pt"))
    if len(existing) >= ENSEMBLE_SIZE:
        log(f"  [{task_name}] Ensemble already trained. Skipping.")
        return
    
    log(f"  [{task_name}] Training {ENSEMBLE_SIZE}× dynamics ensemble...")
    data = np.load(str(DATA_DIR / f"{task_name}.npz"))
    z_t = torch.tensor(data["z_t"], dtype=torch.float32)
    a_t = torch.tensor(data["a_t"], dtype=torch.float32)
    z_next = torch.tensor(data["z_next"], dtype=torch.float32)
    n = len(z_t)
    split = int(n * 0.95)
    
    for ens_idx in range(ENSEMBLE_SIZE):
        log(f"    Ensemble member {ens_idx+1}/{ENSEMBLE_SIZE}...")
        torch.manual_seed(42 + ens_idx * 1000)
        perm = torch.randperm(n)
        
        train_z, train_a, train_zn = z_t[perm[:split]].to(device), a_t[perm[:split]].to(device), z_next[perm[:split]].to(device)
        val_z, val_a, val_zn = z_t[perm[split:]].to(device), a_t[perm[split:]].to(device), z_next[perm[split:]].to(device)
        
        model = DynamicsModel(action_dim=action_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DYNAMICS_EPOCHS)
        best_val = float("inf")
        bs = 4096
        
        for epoch in range(DYNAMICS_EPOCHS):
            model.train()
            total_loss, nb = 0, 0
            p = torch.randperm(len(train_z), device=device)
            for i in range(0, len(train_z), bs):
                idx = p[i:i+bs]
                loss = nn.functional.mse_loss(model(train_z[idx], train_a[idx]), train_zn[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); total_loss += loss.item(); nb += 1
            sched.step()
            
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(val_z, val_a), val_zn).item()
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), str(task_model_dir / f"dynamics_ens_{ens_idx}.pt"))
            if (epoch+1) % 25 == 0:
                log(f"      Ep {epoch+1}/{DYNAMICS_EPOCHS} | T={total_loss/nb:.6f} V={val_loss:.6f} B={best_val:.6f}")
        
        log(f"    Ens {ens_idx} done. Best val: {best_val:.6f}")
        del train_z, train_a, train_zn, val_z, val_a, val_zn, model
        torch.cuda.empty_cache()
    
    log(f"  [{task_name}] Ensemble complete.")


def train_reward_model(task_name, action_dim, device="cuda"):
    task_model_dir = MODEL_DIR / task_name
    task_model_dir.mkdir(parents=True, exist_ok=True)
    
    if (task_model_dir / "reward_model.pt").exists():
        log(f"  [{task_name}] Reward model already trained. Skipping.")
        return
    
    log(f"  [{task_name}] Training reward model...")
    data = np.load(str(DATA_DIR / f"{task_name}.npz"))
    z_t = torch.tensor(data["z_t"], dtype=torch.float32)
    a_t = torch.tensor(data["a_t"], dtype=torch.float32)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32)
    
    n = len(z_t); split = int(n * 0.95)
    perm = torch.randperm(n)
    train_z, train_a, train_r = z_t[perm[:split]].to(device), a_t[perm[:split]].to(device), rewards[perm[:split]].to(device)
    val_z, val_a, val_r = z_t[perm[split:]].to(device), a_t[perm[split:]].to(device), rewards[perm[split:]].to(device)
    
    model = RewardModel(action_dim=action_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_val = float("inf"); bs = 4096
    
    for epoch in range(REWARD_EPOCHS):
        model.train()
        total_loss, nb = 0, 0
        p = torch.randperm(len(train_z), device=device)
        for i in range(0, len(train_z), bs):
            idx = p[i:i+bs]
            loss = nn.functional.mse_loss(model(train_z[idx], train_a[idx]), train_r[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(val_z, val_a), val_r).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(task_model_dir / "reward_model.pt"))
        if (epoch+1) % 10 == 0:
            log(f"    Ep {epoch+1}/{REWARD_EPOCHS} | T={total_loss/nb:.6f} V={val_loss:.6f}")
    
    nonzero = (val_r > 0).float().mean().item()
    log(f"  [{task_name}] Reward model done. Best MSE={best_val:.6f}, nonzero={nonzero:.1%}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("OVERNIGHT v2: Collect → Encode → Train Ensemble → Train Reward")
    log("=" * 70)
    
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    for d in [DATA_DIR, RAW_DIR, MODEL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})")
    overall_start = time.time()
    
    # ── STAGE 1: Collect raw data (CPU-bound, fast) ──────────────
    log("\n" + "=" * 70)
    log("STAGE 1: Raw Data Collection (no GPU needed)")
    log("=" * 70)
    
    for task_name, cfg in TASKS.items():
        collect_raw_data(task_name, cfg)
    
    stage1_time = time.time() - overall_start
    log(f"\nStage 1 complete: {stage1_time/60:.1f} min")
    
    # ── STAGE 2: Batch encode with V-JEPA 2 ──────────────────────
    log("\n" + "=" * 70)
    log("STAGE 2: V-JEPA 2 Batch Encoding (GPU)")
    log("=" * 70)
    
    from transformers import AutoModel
    log("Loading V-JEPA 2...")
    cache_dir = str(BASE_DIR / "cache" / "hf")
    os.makedirs(cache_dir, exist_ok=True)
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device, dtype=torch.float16).eval()
    log("V-JEPA 2 loaded.")
    
    for task_name in TASKS:
        batch_encode_task(task_name, vjepa, device)
    
    del vjepa
    torch.cuda.empty_cache()
    gc.collect()
    
    stage2_time = time.time() - overall_start - stage1_time
    log(f"\nStage 2 complete: {stage2_time/60:.1f} min")
    
    # ── STAGE 3: Ensemble dynamics training ───────────────────────
    log("\n" + "=" * 70)
    log("STAGE 3: Ensemble Dynamics Training")
    log("=" * 70)
    
    for task_name, cfg in TASKS.items():
        train_dynamics_ensemble(task_name, cfg["action_dim"], device)
    
    # ── STAGE 4: Reward model training ────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 4: Reward Model Training")
    log("=" * 70)
    
    for task_name, cfg in TASKS.items():
        train_reward_model(task_name, cfg["action_dim"], device)
    
    # ── Summary ───────────────────────────────────────────────────
    total = time.time() - overall_start
    log("\n" + "=" * 70)
    log("ALL STAGES COMPLETE!")
    log(f"Total: {total/3600:.1f} hrs | Est. cost: ${total/3600 * 0.90:.2f}")
    log("=" * 70)
    
    log("\nData:")
    for f in sorted(DATA_DIR.glob("*.npz")):
        d = np.load(str(f))
        log(f"  {f.name}: {len(d['z_t']):,} transitions ({f.stat().st_size/1e9:.2f} GB)")
    
    log("\nModels:")
    for td in sorted(MODEL_DIR.iterdir()):
        if td.is_dir():
            pts = list(td.glob("*.pt"))
            log(f"  {td.name}/: {len(pts)} models")
    
    summary = {
        "total_hours": total / 3600, "est_cost": total / 3600 * 0.90,
        "tasks": list(TASKS.keys()), "eps_per_task": EPISODES_PER_TASK,
        "ensemble_size": ENSEMBLE_SIZE, "done": datetime.now().isoformat(),
    }
    with open(str(LOG_DIR / "overnight_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    log(f"\nReady for Phase 3 (agent)! 🚀")


if __name__ == "__main__":
    main()
