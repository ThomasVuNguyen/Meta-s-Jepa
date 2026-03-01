"""
Overnight Script v3 — Practical single-GPU approach
====================================================
Inline V-JEPA encoding (no raw frame storage needed).
1000 episodes per task × 3 tasks = ~2 hrs data collection.
Ensemble training = ~1.5 hrs. Total ~4 hrs.

Usage: MUJOCO_GL=osmesa HF_TOKEN=xxx python3 overnight_v3.py
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────

TASKS = {
    "reacher_easy": {"domain": "reacher", "task": "easy", "action_dim": 2, "expert": "reacher"},
    "point_mass_easy": {"domain": "point_mass", "task": "easy", "action_dim": 2, "expert": "point_mass"},
    "cartpole_swingup": {"domain": "cartpole", "task": "swingup", "action_dim": 1, "expert": "cartpole"},
}

EPISODES_PER_TASK = 1000
MAX_STEPS = 200
ENCODE_BATCH = 128
EPSILON = 0.3
LATENT_DIM = 1024
HIDDEN_DIM = 512
ENSEMBLE_SIZE = 5
DYNAMICS_EPOCHS = 100
REWARD_EPOCHS = 50

BASE = Path("/root/vjepa_mvp")
DATA = BASE / "data"
MODELS = BASE / "models"
LOGS = BASE / "logs"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Experts ──────────────────────────────────────────────────────────

def reacher_expert(obs, aspec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(aspec.minimum, aspec.maximum).astype(np.float32)
    return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

def point_mass_expert(obs, aspec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(aspec.minimum, aspec.maximum).astype(np.float32)
    pos = obs.get("position", np.zeros(2))
    vel = obs.get("velocity", np.zeros(2))
    return np.clip(-2.0 * pos - 0.5 * vel, aspec.minimum, aspec.maximum).astype(np.float32)

def cartpole_expert(obs, aspec):
    if np.random.rand() < EPSILON:
        return np.random.uniform(aspec.minimum, aspec.maximum).astype(np.float32)
    pos = obs.get("position", np.zeros(3))
    vel = obs.get("velocity", np.zeros(2))
    cos_a = pos[1] if len(pos) > 1 else 1.0
    sin_a = pos[2] if len(pos) > 2 else 0.0
    cart_pos, cart_vel = pos[0], vel[0] if len(vel) > 0 else 0
    pole_vel = vel[1] if len(vel) > 1 else 0
    angle = np.arctan2(sin_a, cos_a)
    energy = 0.5 * pole_vel**2 + 9.81 * (cos_a - 1)
    if abs(angle) < 0.3:
        a = 5.0*angle + 1.0*pole_vel - 2.0*cart_pos - 1.0*cart_vel
    else:
        a = 3.0*pole_vel*cos_a + 0.5*energy*np.sign(pole_vel)
    return np.clip(np.array([a]), aspec.minimum, aspec.maximum).astype(np.float32)

EXPERTS = {"reacher": reacher_expert, "point_mass": point_mass_expert, "cartpole": cartpole_expert}


# ── V-JEPA Encoder ──────────────────────────────────────────────────

def encode_frames(vjepa, frames, device):
    """Encode all frames using 8-frame sliding windows."""
    from torchvision import transforms
    from PIL import Image
    
    ET = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    ft = [ET(Image.fromarray(f)) for f in frames]
    embs = []
    
    for i in range(0, len(ft), ENCODE_BATCH):
        clips = []
        for t in range(i, min(i + ENCODE_BATCH, len(ft))):
            ws = max(0, t - 7)
            w = ft[ws:t+1]
            while len(w) < 8: w = [w[0]] + w
            clips.append(torch.stack(w))
        
        ct = torch.stack(clips).to(device, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=ct)
            embs.append(out.last_hidden_state.mean(dim=1).cpu().float().numpy())
        del ct
    
    return np.concatenate(embs, axis=0)


# ── Data Collection ─────────────────────────────────────────────────

def collect_and_encode(task_name, cfg, vjepa, device):
    """Collect episodes, encoding each one inline."""
    from dm_control import suite
    
    out_path = DATA / f"{task_name}.npz"
    if out_path.exists():
        d = np.load(str(out_path))
        if len(d["z_t"]) >= EPISODES_PER_TASK * MAX_STEPS * 0.9:
            log(f"  [{task_name}] Already done ({len(d['z_t']):,} transitions). Skipping.")
            return
    
    expert = EXPERTS[cfg["expert"]]
    log(f"  [{task_name}] Collecting + encoding {EPISODES_PER_TASK} episodes...")
    
    all_z, all_a, all_zn, all_r = [], [], [], []
    total_r, successes = 0, 0
    start = time.time()
    
    for ep in range(EPISODES_PER_TASK):
        env = suite.load(cfg["domain"], cfg["task"], task_kwargs={"random": ep})
        aspec = env.action_spec()
        ts = env.reset(); obs = ts.observation
        
        frames = [env.physics.render(height=224, width=224, camera_id=0).copy()]
        actions, rewards = [], []
        
        for _ in range(MAX_STEPS):
            a = expert(obs, aspec)
            actions.append(a.copy())
            ts = env.step(a); obs = ts.observation
            rewards.append(float(ts.reward or 0.0))
            frames.append(env.physics.render(height=224, width=224, camera_id=0).copy())
        
        embs = encode_frames(vjepa, frames, device)
        all_z.append(embs[:-1])
        all_zn.append(embs[1:])
        all_a.append(np.array(actions, dtype=np.float32))
        all_r.append(np.array(rewards, dtype=np.float32))
        
        er = sum(rewards); total_r += er
        if er > 100: successes += 1
        
        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - start
            rate = (ep+1) / (elapsed/60)
            eta = (EPISODES_PER_TASK - ep - 1) / rate
            log(f"    Ep {ep+1:4d}/{EPISODES_PER_TASK} | R={total_r/(ep+1):.1f} | "
                f"SR={successes/(ep+1):.1%} | {rate:.1f} ep/min | ETA {eta:.0f} min")
        
        # Checkpoint every 250 episodes
        if (ep+1) % 250 == 0:
            _save(task_name, all_z, all_a, all_zn, all_r, partial=True)
    
    _save(task_name, all_z, all_a, all_zn, all_r)
    elapsed = time.time() - start
    log(f"  [{task_name}] Done: {len(all_z)*MAX_STEPS:,} transitions in {elapsed/60:.1f} min")


def _save(name, z, a, zn, r, partial=False):
    DATA.mkdir(parents=True, exist_ok=True)
    Z = np.concatenate(z); A = np.concatenate(a); ZN = np.concatenate(zn); R = np.concatenate(r)
    p = DATA / f"{name}.npz"
    np.savez_compressed(str(p), z_t=Z, a_t=A, z_next=ZN, rewards=R)
    log(f"    Saved: {len(Z):,} transitions ({p.stat().st_size/1e9:.2f} GB)")


# ── Models ───────────────────────────────────────────────────────────

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


def train_ensemble(task_name, adim, device):
    md = MODELS / task_name; md.mkdir(parents=True, exist_ok=True)
    if len(list(md.glob("dyn_*.pt"))) >= ENSEMBLE_SIZE:
        log(f"  [{task_name}] Ensemble exists. Skip."); return
    
    log(f"  [{task_name}] Training {ENSEMBLE_SIZE}× dynamics...")
    d = np.load(str(DATA / f"{task_name}.npz"))
    Z = torch.tensor(d["z_t"], dtype=torch.float32)
    A = torch.tensor(d["a_t"], dtype=torch.float32)
    ZN = torch.tensor(d["z_next"], dtype=torch.float32)
    n = len(Z); sp = int(n * 0.95)
    
    for ei in range(ENSEMBLE_SIZE):
        torch.manual_seed(42 + ei*1000)
        p = torch.randperm(n)
        tz, ta, tzn = Z[p[:sp]].to(device), A[p[:sp]].to(device), ZN[p[:sp]].to(device)
        vz, va, vzn = Z[p[sp:]].to(device), A[p[sp:]].to(device), ZN[p[sp:]].to(device)
        
        m = DynamicsModel(adim=adim).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DYNAMICS_EPOCHS)
        best = float("inf"); bs = 4096
        
        for epoch in range(DYNAMICS_EPOCHS):
            m.train(); tl, nb = 0, 0
            pp = torch.randperm(len(tz), device=device)
            for i in range(0, len(tz), bs):
                idx = pp[i:i+bs]
                loss = nn.functional.mse_loss(m(tz[idx], ta[idx]), tzn[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); tl += loss.item(); nb += 1
            sch.step()
            m.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(m(vz, va), vzn).item()
            if vl < best:
                best = vl; torch.save(m.state_dict(), str(md / f"dyn_{ei}.pt"))
            if (epoch+1) % 25 == 0:
                log(f"    E{ei} ep{epoch+1} T={tl/nb:.6f} V={vl:.6f} B={best:.6f}")
        
        log(f"    E{ei} done. Best={best:.6f}")
        del tz, ta, tzn, vz, va, vzn, m; torch.cuda.empty_cache()


def train_reward(task_name, adim, device):
    md = MODELS / task_name; md.mkdir(parents=True, exist_ok=True)
    if (md / "reward.pt").exists():
        log(f"  [{task_name}] Reward exists. Skip."); return
    
    log(f"  [{task_name}] Training reward model...")
    d = np.load(str(DATA / f"{task_name}.npz"))
    Z = torch.tensor(d["z_t"], dtype=torch.float32)
    A = torch.tensor(d["a_t"], dtype=torch.float32)
    R = torch.tensor(d["rewards"], dtype=torch.float32)
    n = len(Z); sp = int(n*0.95); p = torch.randperm(n)
    tz, ta, tr = Z[p[:sp]].to(device), A[p[:sp]].to(device), R[p[:sp]].to(device)
    vz, va, vr = Z[p[sp:]].to(device), A[p[sp:]].to(device), R[p[sp:]].to(device)
    
    m = RewardModel(adim=adim).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    best = float("inf"); bs = 4096
    
    for epoch in range(REWARD_EPOCHS):
        m.train(); tl, nb = 0, 0
        pp = torch.randperm(len(tz), device=device)
        for i in range(0, len(tz), bs):
            idx = pp[i:i+bs]
            loss = nn.functional.mse_loss(m(tz[idx], ta[idx]), tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(vz, va), vr).item()
        if vl < best:
            best = vl; torch.save(m.state_dict(), str(md / "reward.pt"))
        if (epoch+1) % 10 == 0:
            log(f"    ep{epoch+1} T={tl/nb:.6f} V={vl:.6f} B={best:.6f}")
    log(f"  [{task_name}] Reward done. Best={best:.6f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("V3: 1000 eps/task × 3 tasks + ensemble + reward")
    log("=" * 60)
    
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    for d in [DATA, MODELS, LOGS]: d.mkdir(parents=True, exist_ok=True)
    
    device = "cuda"
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    t0 = time.time()
    
    # Load V-JEPA 2
    from transformers import AutoModel
    log("Loading V-JEPA 2 ViT-L...")
    cache = str(BASE / "cache/hf"); os.makedirs(cache, exist_ok=True)
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True,
        cache_dir=cache).to(device, dtype=torch.float16).eval()
    log("V-JEPA 2 loaded.")
    
    # Collect + encode
    log("\n=== DATA COLLECTION + ENCODING ===")
    for tn, cfg in TASKS.items():
        collect_and_encode(tn, cfg, vjepa, device)
    
    del vjepa; torch.cuda.empty_cache(); gc.collect()
    
    # Train
    log("\n=== ENSEMBLE DYNAMICS ===")
    for tn, cfg in TASKS.items():
        train_ensemble(tn, cfg["action_dim"], device)
    
    log("\n=== REWARD MODELS ===")
    for tn, cfg in TASKS.items():
        train_reward(tn, cfg["action_dim"], device)
    
    # Summary
    total = time.time() - t0
    log(f"\n{'='*60}")
    log(f"DONE! {total/3600:.1f} hrs, ~${total/3600 * 1.29:.2f}")
    log(f"{'='*60}")
    
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(str(f))
        log(f"  {f.name}: {len(d['z_t']):,} trans ({f.stat().st_size/1e9:.2f}GB)")
    for td in sorted(MODELS.iterdir()):
        if td.is_dir():
            log(f"  {td.name}/: {len(list(td.glob('*.pt')))} models")
    
    json.dump({"hrs": total/3600, "cost": total/3600*1.29, "done": datetime.now().isoformat()},
              open(str(LOGS / "summary.json"), "w"), indent=2)
    log("Ready for Phase 3! 🚀")

if __name__ == "__main__":
    main()
