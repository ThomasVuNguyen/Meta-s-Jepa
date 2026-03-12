"""
Experiment 11 — 3D Robot Arm: Full Dyna Pipeline (V-JEPA + Learned Reward + Replay Buffer)
===========================================================================================
Adapts the Experiment 10 walker pipeline for a custom 3-DOF robot arm.

Task: Reach a randomly placed target sphere with the end-effector.
Action space: 3-dim continuous [-1, 1] (base yaw, shoulder pitch, elbow pitch).
Reward: 1.0 - tanh(5 * distance) (smooth, peaks at ~1.0).

Pipeline:
  1. Collect 20K offline frames (random policy) → encode with V-JEPA → cache
  2. Train dynamics MLP from scratch
  3. Collect 30 seed rollouts with ground-truth rewards → train reward head
  4. R0: Evaluate baseline (10 MPC vs 10 random, paired seeds)
  5. R1–R4: Dyna loop (collect → encode → fine-tune dynamics → retrain reward → evaluate)

Goal: ≥80% win rate → green light for physical robot build.
"""

import os, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import modal

# ── Modal setup ─────────────────────────────────────────────────────────────
app = modal.App("vjepa-arm3d-dyna")

vol_weights  = modal.Volume.from_name("vjepa2-weights",        create_if_missing=True)
vol_cache    = modal.Volume.from_name("vjepa2-arm3d-cache",    create_if_missing=True)
vol_rollouts = modal.Volume.from_name("vjepa2-arm3d-rollouts", create_if_missing=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf", "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow scipy tqdm imageio[ffmpeg] scikit-learn",
    )
    .add_local_file(os.path.join(_script_dir, "arm3d.xml"), "/app/arm3d.xml")
    .add_local_file(os.path.join(_script_dir, "arm3d_env.py"), "/app/arm3d_env.py")
)

os.environ["MUJOCO_GL"]         = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


# ── Model definitions ────────────────────────────────────────────────────────
class DynamicsMLP(nn.Module):
    def __init__(self, latent_dim=1024, action_dim=3, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))


class RewardMLP(nn.Module):
    """Predicts ground-truth reward from a single latent frame."""
    def __init__(self, latent_dim=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


# ── Distributed V-JEPA Embedding ─────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        "/weights":  vol_weights,
        "/rollouts": vol_rollouts,
    },
    max_containers=10,
)
def embed_chunk(start_idx, end_idx, frames_path="/rollouts/arm3d/frames.npy"):
    import os
    import numpy as np
    import torch
    import PIL.Image
    from transformers import AutoVideoProcessor, AutoModel

    os.environ["HF_HOME"] = "/weights/hf"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc  = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256", cache_dir="/weights/hf")
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", cache_dir="/weights/hf",
                                       torch_dtype=torch.float16).to(device)
    model.eval()

    IMG_SIZE = 256
    N_FRAMES = 8

    print(f"[{start_idx}:{end_idx}] Loading frames from disk...")
    frames_np = np.load(frames_path, mmap_mode='r')
    chunk = frames_np[start_idx:end_idx]

    embs = []
    bs = 8
    print(f"[{start_idx}:{end_idx}] Embedding {len(chunk)} frames...")
    for s in range(0, len(chunk), bs):
        batch = chunk[s:s+bs]
        videos = []
        for f in batch:
            img = PIL.Image.fromarray(f).resize((IMG_SIZE, IMG_SIZE))
            videos.append([img] * N_FRAMES)

        inp = proc(videos=videos, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
            embs.append(out.last_hidden_state.mean(dim=1).cpu().float())

        if (s + len(batch)) % 160 == 0:
            print(f"[{start_idx}:{end_idx}] Embedded {s + len(batch)} / {len(chunk)} frames...")

    print(f"[{start_idx}:{end_idx}] Done!")
    return torch.cat(embs, dim=0)


# ── Main experiment function ─────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={
        "/weights":  vol_weights,
        "/cache":    vol_cache,
        "/rollouts": vol_rollouts,
    },
)
def arm3d_dyna_experiment():
    import os, sys, time, math, random, json
    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoVideoProcessor, AutoModel
    import PIL.Image

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["TRANSFORMERS_CACHE"]= "/weights/hf"

    # Import our custom environment
    sys.path.insert(0, "/app")
    from arm3d_env import Arm3DEnv

    device = torch.device("cuda")

    # ── Hyperparameters ──────────────────────────────────────────────────
    LATENT_DIM  = 1024
    ACTION_DIM  = 3
    N_FRAMES    = 8
    IMG_SIZE    = 256
    EP_STEPS    = 100

    # Data collection
    N_OFFLINE_EPS = 200       # 200 episodes × 100 steps = 20K transitions
    N_SEED_EPS    = 30        # Seed rollouts for initial reward head training

    # Dyna loop
    ROLLOUTS_PER_ROUND = 60
    N_DYNA_ROUNDS = 4         # Extra round for stability check

    # Training
    DYN_LR      = 3e-4
    DYN_EPOCHS  = 100         # Initial training (from scratch)
    FT_LR       = 5e-5
    FT_EPOCHS   = 15          # Fine-tuning per Dyna round
    RW_LR       = 1e-3
    RW_EPOCHS   = 20

    # CEM planning
    CEM_H       = 8           # Shorter horizon — reaching is simpler
    CEM_N       = 256         # Fewer candidates — 3D action space is simpler
    CEM_K       = 8
    CEM_ITERS   = 5

    # Evaluation
    EVAL_EPS    = 10

    print("=" * 70)
    print("EXPERIMENT 11: 3D Robot Arm — Full Dyna Pipeline")
    print("=" * 70)

    # ── [1] Load V-JEPA 2 encoder ────────────────────────────────────────
    print("\n[1] Loading V-JEPA 2 encoder...")
    proc  = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                                                cache_dir="/weights/hf")
    vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                                       cache_dir="/weights/hf").to(device).eval()

    def embed_frames(frames_np):
        """frames_np: list of H×W×3 uint8 arrays → (1, latent_dim)"""
        pils = [PIL.Image.fromarray(f).resize((IMG_SIZE, IMG_SIZE)) for f in frames_np]
        inp  = proc(videos=[pils], return_tensors="pt").to(device)
        with torch.no_grad():
            out = vjepa(**inp)
        return out.last_hidden_state.mean(dim=1)

    def embed_batch(frames_np, bs=8):
        embs = []
        for s in range(0, len(frames_np), bs):
            batch = frames_np[s:s+bs]
            videos = []
            for f in batch:
                img = PIL.Image.fromarray(f).resize((IMG_SIZE, IMG_SIZE))
                videos.append([img] * N_FRAMES)

            inp = proc(videos=videos, return_tensors="pt").to(device)
            with torch.no_grad():
                out = vjepa(**inp)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().float())
            if (s + len(batch)) % 80 == 0:
                print(f"        Embedded {s+len(batch)} / {len(frames_np)} frames...", flush=True)
        return torch.cat(embs, dim=0)

    def collect_episode(env, policy_fn, ep_steps, record=False):
        """Run one episode; policy_fn(latent) → action array.
        Returns dict with latents, actions, next_latents, rewards.
        If record=True, also returns 'video_frames' list of uint8 arrays."""
        frame = env.reset()
        video_frames = [frame.copy()] if record else None
        frames = [frame] * N_FRAMES
        z = embed_frames(frames)
        zs, acts, next_zs, rewards = [], [], [], []
        for _ in range(ep_steps):
            with torch.no_grad():
                a_t = policy_fn(z)
            frame, reward, done, info = env.step(a_t)
            if record:
                video_frames.append(frame.copy())
            frames = frames[1:] + [frame]
            z_next = embed_frames(frames)
            zs.append(z.cpu())
            acts.append(torch.tensor(a_t, dtype=torch.float32).unsqueeze(0))
            next_zs.append(z_next.cpu())
            rewards.append(float(reward))
            z = z_next
        result = {
            "z":      torch.cat(zs, dim=0),
            "a":      torch.cat(acts, dim=0),
            "z_next": torch.cat(next_zs, dim=0),
            "r":      torch.tensor(rewards, dtype=torch.float32),
        }
        if record:
            result["video_frames"] = video_frames
        return result

    def random_policy(z):
        return np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)

    # CEM planning with learned reward
    def cem_policy(z, dyn_model, rew_model):
        z_t = z.to(device)
        mu  = torch.zeros(CEM_H, ACTION_DIM, device=device)
        std = torch.ones(CEM_H, ACTION_DIM, device=device) * 0.5
        for _ in range(CEM_ITERS):
            acts = (mu.unsqueeze(0) + std.unsqueeze(0) *
                    torch.randn(CEM_N, CEM_H, ACTION_DIM, device=device)).clamp(-1, 1)
            total_r = torch.zeros(CEM_N, device=device)
            z_sim   = z_t.expand(CEM_N, -1)
            for h in range(CEM_H):
                a_h     = acts[:, h, :]
                z_sim   = dyn_model(z_sim, a_h)
                r_hat   = rew_model(z_sim)
                total_r = total_r + r_hat
            top_k   = total_r.topk(CEM_K).indices
            mu      = acts[top_k].mean(0)
            std     = acts[top_k].std(0).clamp(1e-3, 2.0)
        return mu[0].cpu().numpy().astype(np.float32)

    # ── [2] Collect offline dataset ──────────────────────────────────────
    print(f"\n[2] Collecting {N_OFFLINE_EPS} offline episodes (random policy)...")
    offline_cache = "/rollouts/arm3d/frames.npy"
    actions_cache = "/rollouts/arm3d/actions.npy"
    rewards_cache = "/rollouts/arm3d/rewards.npy"
    embed_cache   = "/cache/arm3d_embeddings_cache.pt"

    os.makedirs("/rollouts/arm3d", exist_ok=True)

    if os.path.exists(embed_cache):
        print("    Found existing embeddings cache! Loading...")
        cache = torch.load(embed_cache, map_location="cpu")
        Z_offline  = cache["z"]
        Zn_offline = cache["z_next"]
        A_offline  = cache["a"]
        R_offline  = cache["r"]
        print(f"    Loaded {len(Z_offline)} offline transitions.")
    else:
        if os.path.exists(offline_cache):
            print("    Found existing raw frames, loading...")
            all_frames  = np.load(offline_cache)
            all_actions = np.load(actions_cache)
            all_rewards = np.load(rewards_cache)
        else:
            print("    Collecting fresh offline data...")
            t0 = time.time()
            all_frames_list  = []
            all_actions_list = []
            all_rewards_list = []

            for ep_i in range(N_OFFLINE_EPS):
                env = Arm3DEnv(xml_path="/app/arm3d.xml", seed=ep_i)
                frame = env.reset()
                all_frames_list.append(frame)

                for step in range(EP_STEPS):
                    action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
                    frame, reward, done, info = env.step(action)
                    all_frames_list.append(frame)
                    all_actions_list.append(action)
                    all_rewards_list.append(reward)

                env.close()
                if (ep_i + 1) % 50 == 0:
                    print(f"      Collected {ep_i+1}/{N_OFFLINE_EPS} episodes "
                          f"({(time.time()-t0)/60:.1f} min)")

            all_frames  = np.array(all_frames_list, dtype=np.uint8)
            all_actions = np.array(all_actions_list, dtype=np.float32)
            all_rewards = np.array(all_rewards_list, dtype=np.float32)

            np.save(offline_cache, all_frames)
            np.save(actions_cache, all_actions)
            np.save(rewards_cache, all_rewards)
            vol_rollouts.commit()
            print(f"    Saved {len(all_frames)} frames in {(time.time()-t0)/60:.1f} min")

        # Embed all frames with distributed workers
        print(f"    Embedding {len(all_frames)} frames across parallel workers...")
        t0 = time.time()

        chunk_size = 2000
        ranges = [(i, min(i+chunk_size, len(all_frames)))
                  for i in range(0, len(all_frames), chunk_size)]

        results = list(embed_chunk.starmap(ranges))
        Z_all = torch.cat(results, dim=0)
        print(f"    Embedded {len(Z_all)} frames in {(time.time()-t0)/60:.1f} min")

        # Build transition dataset
        # Frames: [f0, f1, f2, ...] where f0 is reset frame of ep0
        # We need to handle episode boundaries: each ep has EP_STEPS+1 frames
        # and EP_STEPS transitions
        Z_offline_list  = []
        Zn_offline_list = []
        A_offline_list  = []
        R_offline_list  = []

        for ep_i in range(N_OFFLINE_EPS):
            start = ep_i * (EP_STEPS + 1)
            for step in range(EP_STEPS):
                Z_offline_list.append(Z_all[start + step])
                Zn_offline_list.append(Z_all[start + step + 1])

        Z_offline  = torch.stack(Z_offline_list)
        Zn_offline = torch.stack(Zn_offline_list)
        A_offline  = torch.tensor(all_actions, dtype=torch.float32)
        R_offline  = torch.tensor(all_rewards, dtype=torch.float32)

        # Save to cache
        torch.save({"z": Z_offline, "z_next": Zn_offline, "a": A_offline, "r": R_offline},
                   embed_cache)
        vol_cache.commit()
        print(f"    Cached {len(Z_offline)} offline transitions")

    print(f"    Offline dataset: {len(Z_offline)} transitions")

    # ── [3] Train initial dynamics model ─────────────────────────────────
    print("\n[3] Training initial dynamics model from scratch...")
    dyn_model = DynamicsMLP(LATENT_DIM, ACTION_DIM).to(device)
    rew_model = RewardMLP(LATENT_DIM).to(device)

    dyn_ckpt = "/cache/arm3d_dynamics_initial.pt"
    if os.path.exists(dyn_ckpt):
        print("    Loading cached dynamics model...")
        dyn_model.load_state_dict(torch.load(dyn_ckpt, map_location=device))
    else:
        t0 = time.time()
        ds = TensorDataset(Z_offline, A_offline, Zn_offline)
        dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(dyn_model.parameters(), lr=DYN_LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DYN_EPOCHS)

        dyn_model.train()
        for ep in range(DYN_EPOCHS):
            for zb, ab, znb in dl:
                zb, ab, znb = zb.to(device), ab.to(device), znb.to(device)
                opt.zero_grad()
                pred = dyn_model(zb, ab)
                loss = nn.functional.mse_loss(pred, znb)
                loss.backward()
                nn.utils.clip_grad_norm_(dyn_model.parameters(), 1.0)
                opt.step()
            scheduler.step()
            if (ep + 1) % 20 == 0:
                n_val = max(1, len(Z_offline) // 10)
                with torch.no_grad():
                    zv = Z_offline[-n_val:].to(device)
                    av = A_offline[-n_val:].to(device)
                    znv = Zn_offline[-n_val:].to(device)
                    val = nn.functional.mse_loss(dyn_model(zv, av), znv).item()
                print(f"      Epoch {ep+1}: val_loss={val:.6f}")

        torch.save(dyn_model.state_dict(), dyn_ckpt)
        vol_cache.commit()
        print(f"    Dynamics trained in {(time.time()-t0)/60:.1f} min")

    # ── Helper: fine-tune dynamics on mixed buffer ────────────────────────
    def finetune_dynamics(dyn_model, Z, A, Zn, lr, epochs):
        ds  = TensorDataset(Z, A, Zn)
        dl  = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(dyn_model.parameters(), lr=lr, weight_decay=1e-5)
        dyn_model.train()
        for ep in range(epochs):
            for zb, ab, znb in dl:
                zb, ab, znb = zb.to(device), ab.to(device), znb.to(device)
                opt.zero_grad()
                pred = dyn_model(zb, ab)
                loss = nn.functional.mse_loss(pred, znb)
                loss.backward()
                nn.utils.clip_grad_norm_(dyn_model.parameters(), 1.0)
                opt.step()
        n_val = max(1, len(Z) // 10)
        with torch.no_grad():
            zv, av, znv = Z[-n_val:].to(device), A[-n_val:].to(device), Zn[-n_val:].to(device)
            val_loss = nn.functional.mse_loss(dyn_model(zv, av), znv).item()
        return val_loss

    # ── Helper: train reward head ────────────────────────────────────────
    def train_reward_head(rew_model, Z_rw, R_rw, lr, epochs):
        if len(Z_rw) == 0:
            print("    ⚠ No reward labels — skipping reward head training")
            return float("nan")
        r_mean, r_std = R_rw.mean().item(), R_rw.std().item() + 1e-6
        R_norm = (R_rw - r_mean) / r_std
        ds = TensorDataset(Z_rw, R_norm)
        dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(rew_model.parameters(), lr=lr, weight_decay=1e-4)
        rew_model.train()
        for ep in range(epochs):
            for zb, rb in dl:
                zb, rb = zb.to(device), rb.to(device)
                opt.zero_grad()
                r_hat = rew_model(zb)
                loss = nn.functional.mse_loss(r_hat, rb)
                loss.backward()
                opt.step()
        n_val = max(1, len(Z_rw) // 10)
        with torch.no_grad():
            zv, rv = Z_rw[-n_val:].to(device), R_norm[-n_val:].to(device)
            val_r = nn.functional.mse_loss(rew_model(zv), rv).item()
        rew_model.eval()
        return val_r

    # ── [4] Collect seed rollouts with reward labels ─────────────────────
    print(f"\n[4] Collecting {N_SEED_EPS} seed rollouts with ground-truth rewards...")
    t0 = time.time()
    Z_rw, A_rw, Zn_rw, R_rw = [], [], [], []

    for i in range(N_SEED_EPS):
        env = Arm3DEnv(xml_path="/app/arm3d.xml", seed=10000 + i)
        ep = collect_episode(env, random_policy, EP_STEPS)
        Z_rw.append(ep["z"]);  A_rw.append(ep["a"])
        Zn_rw.append(ep["z_next"]); R_rw.append(ep["r"])
        env.close()
        if (i + 1) % 10 == 0:
            print(f"      Seed episode {i+1}/{N_SEED_EPS} ({(time.time()-t0)/60:.1f} min)")

    Z_rw  = torch.cat(Z_rw,  dim=0)
    A_rw  = torch.cat(A_rw,  dim=0)
    Zn_rw = torch.cat(Zn_rw, dim=0)
    R_rw  = torch.cat(R_rw,  dim=0)
    print(f"    Seed rollouts: {len(Z_rw)} labelled transitions in {(time.time()-t0)/60:.1f} min")

    # Build mixed buffer: offline (permanent) + seed
    Z  = torch.cat([Z_offline,  Z_rw],  dim=0)
    A  = torch.cat([A_offline,  A_rw],  dim=0)
    Zn = torch.cat([Zn_offline, Zn_rw], dim=0)

    # ── [5] Train initial reward head ────────────────────────────────────
    print("\n[5] Training initial reward head...")
    t0 = time.time()
    val_rw = train_reward_head(rew_model, Z_rw, R_rw, RW_LR, RW_EPOCHS)
    print(f"    Reward head trained in {(time.time()-t0)/60:.1f} min  val_loss={val_rw:.4f}")

    # ── Helper: save video from frames ────────────────────────────────────
    def save_video(frames_list, path, fps=20):
        """Save list of uint8 RGB frames as an MP4 video."""
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        for f in frames_list:
            writer.append_data(f)
        writer.close()
        print(f"      📹 Saved video: {path} ({len(frames_list)} frames)")

    def save_side_by_side_video(mpc_frames, rand_frames, path, fps=20):
        """Create side-by-side MPC vs Random comparison video."""
        import imageio
        n = min(len(mpc_frames), len(rand_frames))
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        for i in range(n):
            combined = np.concatenate([mpc_frames[i], rand_frames[i]], axis=1)
            writer.append_data(combined)
        writer.close()
        print(f"      📹 Saved side-by-side: {path}")

    # ── Helper: evaluate one round ───────────────────────────────────────
    def evaluate(dyn_model, rew_model, n_eps, ep_steps, seed_offset=0, round_id=0):
        dyn_model.eval(); rew_model.eval()
        mpc_rewards, rand_rewards = [], []
        wins = 0
        video_dir = f"/cache/videos/round_{round_id}"
        os.makedirs(video_dir, exist_ok=True)

        for ep_i in range(n_eps):
            # MPC episode (with video recording)
            env_mpc = Arm3DEnv(xml_path="/app/arm3d.xml",
                               seed=seed_offset + ep_i * 2)
            ep_mpc = collect_episode(
                env_mpc, lambda z: cem_policy(z, dyn_model, rew_model),
                ep_steps, record=True)
            env_mpc.close()

            # Random episode (with video recording)
            env_rand = Arm3DEnv(xml_path="/app/arm3d.xml",
                                seed=seed_offset + ep_i * 2 + 1)
            ep_rand = collect_episode(env_rand, random_policy, ep_steps, record=True)
            env_rand.close()

            r_mpc  = ep_mpc["r"].sum().item()
            r_rand = ep_rand["r"].sum().item()
            win    = r_mpc > r_rand
            wins  += int(win)
            mpc_rewards.append(r_mpc)
            rand_rewards.append(r_rand)
            mark = "✅" if win else "❌"
            print(f"      ep{ep_i+1}: MPC={r_mpc:.1f}  rand={r_rand:.1f}  {mark}")

            # Save videos: individual + side-by-side
            save_video(ep_mpc["video_frames"],
                      f"{video_dir}/ep{ep_i+1}_mpc.mp4")
            save_video(ep_rand["video_frames"],
                      f"{video_dir}/ep{ep_i+1}_random.mp4")
            save_side_by_side_video(
                ep_mpc["video_frames"], ep_rand["video_frames"],
                f"{video_dir}/ep{ep_i+1}_comparison.mp4")

        vol_cache.commit()
        avg_mpc  = np.mean(mpc_rewards)
        avg_rand = np.mean(rand_rewards)
        pct      = wins / n_eps * 100
        return avg_mpc, avg_rand, pct, mpc_rewards, rand_rewards

    # ── [6] Round 0: Baseline evaluation ─────────────────────────────────
    print("\n--- Round 0: Baseline Eval (initial dynamics + reward head) ---")
    t0 = time.time()
    r0_mpc, r0_rand, r0_win, _, _ = evaluate(
        dyn_model, rew_model, EVAL_EPS, EP_STEPS, round_id=0)
    print(f"  R0: MPC={r0_mpc:.2f}  rand={r0_rand:.2f}  "
          f"win={r0_win:.0f}%  [{(time.time()-t0)/60:.1f}min]")

    results = {
        "experiment": 11,
        "env": "arm3d-reach",
        "approach": "learned reward + dyna + mixed replay buffer",
        "n_dyna_rounds": N_DYNA_ROUNDS,
        "rollouts_per_round": ROLLOUTS_PER_ROUND,
        "rounds": [{
            "round": 0,
            "n_train": len(Z),
            "avg_mpc_reward": r0_mpc,
            "avg_rand_reward": r0_rand,
            "pct_better": r0_win,
            "reward_head_val": val_rw,
        }],
    }

    # ── [7] Dyna rounds ──────────────────────────────────────────────────
    for rnd in range(1, N_DYNA_ROUNDS + 1):
        print(f"\n--- Round {rnd}: Collect {ROLLOUTS_PER_ROUND} on-policy rollouts ---")
        t0 = time.time()
        Z_new, A_new, Zn_new, R_new = [], [], [], []

        for i in range(ROLLOUTS_PER_ROUND):
            seed_i = 20000 + rnd * 1000 + i
            env_i = Arm3DEnv(xml_path="/app/arm3d.xml", seed=seed_i)
            ep = collect_episode(
                env_i, lambda z: cem_policy(z, dyn_model, rew_model), EP_STEPS)
            Z_new.append(ep["z"]);  A_new.append(ep["a"])
            Zn_new.append(ep["z_next"]); R_new.append(ep["r"])
            env_i.close()
            if (i + 1) % 20 == 0:
                print(f"      Collected {i+1}/{ROLLOUTS_PER_ROUND} rollouts "
                      f"({(time.time()-t0)/60:.1f} min)")

        Z_new  = torch.cat(Z_new,  dim=0)
        A_new  = torch.cat(A_new,  dim=0)
        Zn_new = torch.cat(Zn_new, dim=0)
        R_new  = torch.cat(R_new,  dim=0)

        # Append to cumulative mixed buffer (offline permanently retained)
        Z  = torch.cat([Z,  Z_new],  dim=0)
        A  = torch.cat([A,  A_new],  dim=0)
        Zn = torch.cat([Zn, Zn_new], dim=0)
        Z_rw  = torch.cat([Z_rw,  Z_new],  dim=0)
        R_rw  = torch.cat([R_rw,  R_new],  dim=0)
        print(f"    Collected {len(Z_new)} on-policy transitions in {(time.time()-t0)/60:.1f} min")
        print(f"    Total dynamics dataset: {len(Z)} (mixed buffer)")
        print(f"    Total reward dataset:   {len(Z_rw)} labelled")

        # Fine-tune dynamics on mixed buffer
        print(f"    Fine-tuning dynamics for {FT_EPOCHS} epochs (lr={FT_LR})...")
        t1 = time.time()
        dyn_model.train()
        val_dyn = finetune_dynamics(dyn_model, Z, A, Zn, FT_LR, FT_EPOCHS)
        print(f"    Dynamics FT done in {(time.time()-t1)/60:.1f} min  val_loss={val_dyn:.4f}")

        # Re-train reward head on ALL labelled data
        print(f"    Re-training reward head for {RW_EPOCHS} epochs (lr={RW_LR})...")
        t1 = time.time()
        # Reset reward head to avoid catastrophic fits
        rew_model = RewardMLP(LATENT_DIM).to(device)
        val_rw = train_reward_head(rew_model, Z_rw, R_rw, RW_LR, RW_EPOCHS)
        print(f"    Reward head done in {(time.time()-t1)/60:.1f} min  val={val_rw:.4f}")

        # Evaluate
        print(f"    Evaluating round {rnd}...")
        avg_mpc, avg_rand, pct, _, _ = evaluate(
            dyn_model, rew_model, EVAL_EPS, EP_STEPS, seed_offset=rnd * 100, round_id=rnd)
        print(f"  R{rnd}: MPC={avg_mpc:.2f}  rand={avg_rand:.2f}  win={pct:.0f}%  "
              f"dyn_val={val_dyn:.4f}  rw_val={val_rw:.4f}")

        results["rounds"].append({
            "round": rnd,
            "n_train_dynamics": len(Z),
            "n_train_reward":   len(Z_rw),
            "avg_mpc_reward": avg_mpc,
            "avg_rand_reward": avg_rand,
            "pct_better": pct,
            "dyn_val_loss":    val_dyn,
            "reward_head_val": val_rw,
        })

        # Save checkpoints
        torch.save(dyn_model.state_dict(), f"/cache/arm3d_dyn_r{rnd}.pt")
        torch.save(rew_model.state_dict(), f"/cache/arm3d_reward_head_r{rnd}.pt")
        vol_cache.commit()

    # ── [8] Generate result charts ───────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds_list = [r["round"] for r in results["rounds"]]
    mpc_vals    = [r["avg_mpc_reward"] for r in results["rounds"]]
    rnd_vals    = [r["avg_rand_reward"] for r in results["rounds"]]
    wins        = [r["pct_better"] for r in results["rounds"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exp 11 — 3D Robot Arm: Dyna Pipeline Results",
                 fontsize=13, fontweight="bold")

    # Win rate
    axes[0].plot(rounds_list, wins, "o-", color="#4CAF50", lw=2, marker="D")
    axes[0].axhline(80, color="red", ls="--", alpha=0.5, label="80% target")
    axes[0].axhline(50, color="gray", ls=":", alpha=0.5, label="Random parity")
    axes[0].set_xlabel("Dyna Round"); axes[0].set_ylabel("Win % (vs random)")
    axes[0].set_title("Win Rate"); axes[0].set_ylim(0, 105)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # MPC vs random reward
    axes[1].plot(rounds_list, mpc_vals, "o-", color="#2196F3", lw=2, label="MPC")
    axes[1].plot(rounds_list, rnd_vals, "s--", color="#9E9E9E", lw=2, label="Random")
    axes[1].set_xlabel("Dyna Round"); axes[1].set_ylabel("Avg Episode Reward")
    axes[1].set_title("Reward: MPC vs Random"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Model losses
    dyn_vals = [r.get("dyn_val_loss", None) for r in results["rounds"]]
    rw_vals  = [r.get("reward_head_val", None) for r in results["rounds"]]
    valid_dyn = [(r, v) for r, v in zip(rounds_list, dyn_vals) if v is not None]
    valid_rw  = [(r, v) for r, v in zip(rounds_list, rw_vals) if v is not None]
    if valid_dyn:
        axes[2].plot([r for r, v in valid_dyn], [v for r, v in valid_dyn],
                     "o-", color="#FF9800", lw=2, label="Dynamics val")
    if valid_rw:
        axes[2].plot([r for r, v in valid_rw], [v for r, v in valid_rw],
                     "s-", color="#9C27B0", lw=2, label="Reward val")
    axes[2].set_xlabel("Dyna Round"); axes[2].set_ylabel("Val Loss")
    axes[2].set_title("Model Losses"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    chart_path = "/cache/arm3d_dyna_results.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight"); plt.close()

    # Save results JSON
    print(json.dumps(results, indent=2))
    with open("/cache/arm3d_dyna_results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol_cache.commit()

    print("\n=== EXPERIMENT 11 COMPLETE ===")
    for r in results["rounds"]:
        rw_v = r.get("reward_head_val", float("nan"))
        dyn_v = r.get("dyn_val_loss", "---")
        print(f"  R{r['round']}: MPC={r['avg_mpc_reward']:.2f}  "
              f"rand={r['avg_rand_reward']:.2f}  win={r['pct_better']:.0f}%  "
              f"dyn_val={dyn_v}  rw_val={rw_v:.4f}")

    print("\n  ✓ arm3d_dyna_results.json")
    print("  ✓ arm3d_dyna_results.png")
    print("  ✓ arm3d_dyn_rN.pt / arm3d_reward_head_rN.pt")

    # Return summary for easy checking
    best_win = max(r["pct_better"] for r in results["rounds"])
    print(f"\n  🏆 BEST WIN RATE: {best_win:.0f}%")
    if best_win >= 80:
        print("  ✅ GREEN LIGHT: Proceed to physical robot build!")
    elif best_win >= 60:
        print("  🟡 PROMISING: Pipeline works, may need more rounds or tuning")
    else:
        print("  🔴 NEEDS WORK: Pipeline improvements needed before hardware")


@app.local_entrypoint()
def main():
    arm3d_dyna_experiment.remote()
