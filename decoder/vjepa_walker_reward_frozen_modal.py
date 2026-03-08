"""
Experiment 9 — Walker-Walk: Frozen Task-Aligned Reward Head
=======================================================================
Root cause of Exp 8's R2 collapse: The reward head overfit to the narrowing
on-policy distribution, causing planning to fail.

Fix:
  1. Train reward head normally on seed data and R1.
  2. FREEZE the reward head for all subsequent rounds (R2, R3).
  3. Run 3 Dyna rounds total to verify long-term stability.

Expected outcome: win-rate holds steady at 80% across multiple rounds.
"""

import os, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import modal

# ── Modal setup ─────────────────────────────────────────────────────────────
app = modal.App("vjepa-walker-reward-frozen-dyna")

# Volumes
vol_weights  = modal.Volume.from_name("vjepa2-weights",        create_if_missing=True)
vol_cache    = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
vol_rollouts = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf", "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow scipy tqdm imageio[ffmpeg] scikit-learn",
    )
)

os.environ["MUJOCO_GL"]        = "osmesa"
os.environ["PYOPENGL_PLATFORM"]= "osmesa"

# ── Model definitions ────────────────────────────────────────────────────────
class DynamicsMLP(nn.Module):
    def __init__(self, latent_dim=1024, action_dim=6, hidden=512):
        super().__init__()
        # Exact Exp6 architecture: 3 linear layers
        # net.0=Linear(latent+action,hidden), net.1=LN, net.2=GELU
        # net.3=Linear(hidden,hidden),         net.4=LN, net.5=GELU
        # net.6=Linear(hidden,latent_dim)      [final projection, no activation]
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))
class RewardMLP(nn.Module):
    """Predicts ground-truth dm_control reward from a single latent frame."""
    def __init__(self, latent_dim=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


# ── Main experiment function ─────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/weights":  vol_weights,
        "/cache":    vol_cache,
        "/rollouts": vol_rollouts,
    },
)
def walker_reward_dyna_experiment():
    import os, time, math, random, json
    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoVideoProcessor, AutoModel
    import PIL.Image, imageio

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["TRANSFORMERS_CACHE"]= "/weights/hf"

    from dm_control import suite

    device = torch.device("cuda")
    LATENT_DIM  = 1024
    ACTION_DIM  = 6
    N_FRAMES    = 8
    IMG_SIZE    = 256
    ROLLOUTS_PER_ROUND = 60   # more rollouts for reward signal coverage
    FT_LR       = 5e-5
    FT_EPOCHS   = 15
    RW_LR       = 1e-3
    RW_EPOCHS   = 20
    N_DYNA_ROUNDS = 3
    EVAL_EPS    = 10
    EP_STEPS    = 100
    CEM_H       = 15   # planning horizon
    CEM_N       = 512  # CEM candidates
    CEM_K       = 8    # CEM elites
    CEM_ITERS   = 3

    print("=" * 70)
    print("META-S-JEPA  Experiment 9: Walker-Walk Frozen Reward Head")
    print("=" * 70)

    # ── [1] Load V-JEPA 2 encoder ──────────────────────────────────────────
    print("\n[1] Loading V-JEPA 2 encoder...")
    proc  = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                                                cache_dir="/weights/hf")
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                                       cache_dir="/weights/hf").to(device).eval()

    def embed_frames(frames_np):
        """frames_np: list of H×W×3 uint8 arrays → (1, latent_dim)"""
        pils = [PIL.Image.fromarray(f).resize((IMG_SIZE, IMG_SIZE)) for f in frames_np]
        inp  = proc(videos=[pils], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        return out.last_hidden_state.mean(dim=1)   # (1, latent_dim)

    def render_env(env):
        return env.physics.render(height=IMG_SIZE, width=IMG_SIZE, camera_id=0)

    def collect_episode(env, policy_fn, ep_steps):
        """Run one episode; policy_fn(latent) → action array.
        Returns dict with latents, actions, next_latents, rewards, pixels."""
        ts = env.reset()
        frames = [render_env(env)] * N_FRAMES
        z = embed_frames(frames)
        zs, acts, next_zs, rewards = [], [], [], []
        for _ in range(ep_steps):
            with torch.no_grad():
                a_t = policy_fn(z)
            ts = env.step(a_t)
            frames = frames[1:] + [render_env(env)]
            z_next = embed_frames(frames)
            zs.append(z.cpu())
            acts.append(torch.tensor(a_t, dtype=torch.float32).unsqueeze(0))
            next_zs.append(z_next.cpu())
            rewards.append(float(ts.reward if ts.reward is not None else 0.0))
            z = z_next
        return {
            "z":      torch.cat(zs, dim=0),
            "a":      torch.cat(acts, dim=0),
            "z_next": torch.cat(next_zs, dim=0),
            "r":      torch.tensor(rewards, dtype=torch.float32),
        }

    def random_policy(z):
        return np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)

    # CEM planning using a custom reward function
    def cem_policy(z, dyn_model, rew_model):
        z_t = z.to(device)
        mu  = torch.zeros(CEM_H, ACTION_DIM, device=device)
        std = torch.ones(CEM_H,  ACTION_DIM, device=device)
        for _ in range(CEM_ITERS):
            acts = (mu.unsqueeze(0) + std.unsqueeze(0) *
                    torch.randn(CEM_N, CEM_H, ACTION_DIM, device=device)).clamp(-1, 1)
            total_r = torch.zeros(CEM_N, device=device)
            z_sim   = z_t.expand(CEM_N, -1)
            for h in range(CEM_H):
                a_h     = acts[:, h, :]
                z_sim   = dyn_model(z_sim, a_h)
                r_hat   = rew_model(z_sim)           # ← task-aligned reward!
                total_r = total_r + r_hat
            top_k   = total_r.topk(CEM_K).indices
            mu      = acts[top_k].mean(0)
            std     = acts[top_k].std(0).clamp(1e-3, 2.0)
        return mu[0].cpu().numpy().astype(np.float32)

    # ── [2] Load Exp6 offline walker data (latents already embedded) ────────
    print("\n[2] Loading offline walker embeddings from Exp6 cache...")
    t0 = time.time()
    cache_file = "/cache/embeddings_cache_walker.pt"
    if os.path.exists(cache_file):
        cache = torch.load(cache_file, map_location="cpu")
        Z  = cache["z"]
        A  = cache["a"]
        Zn = cache["z_next"]
        R  = cache.get("r", None)
        print(f"    Loaded {len(Z)} transitions from cache.")
        if R is None:
            print("    ⚠ No reward labels in cache — will collect fresh labelled data.")
            R = torch.zeros(len(Z))   # placeholder; reward head won't use these
    else:
        print("    No cached embeddings found. Will collect from scratch.")
        R = torch.zeros(0)
        Z  = torch.zeros(0, LATENT_DIM)
        A  = torch.zeros(0, ACTION_DIM)
        Zn = torch.zeros(0, LATENT_DIM)

    print(f"    Offline dataset: {len(Z)} transitions  (loaded in {(time.time()-t0)/60:.1f} min)")

    # ── [3] Load Exp6 dynamics MLP ─────────────────────────────────────────
    print("\n[3] Loading Exp6 walker dynamics MLP checkpoint (warm start)...")
    dyn_model = DynamicsMLP(LATENT_DIM, ACTION_DIM).to(device)
    ckpt_path = "/cache/walker_dynamics_mlp.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # checkpoint may be bare state_dict or a wrapper dict
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        dyn_model.load_state_dict(state)
        val_info = ckpt.get("final_val_loss", "?") if isinstance(ckpt, dict) else "?"
        print(f"    ✓ Loaded walker_dynamics_mlp.pt  val_loss={val_info}")
    else:
        print("    ⚠ checkpoint not found, starting fresh")

    rew_model = RewardMLP(LATENT_DIM).to(device)

    # ── Helper: train dynamics MLP (warm-start fine-tune) ──────────────────
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
                opt.step()
        # val loss on 10% held-out
        n_val = max(1, len(Z) // 10)
        with torch.no_grad():
            zv, av, znv = Z[-n_val:].to(device), A[-n_val:].to(device), Zn[-n_val:].to(device)
            val_loss = nn.functional.mse_loss(dyn_model(zv, av), znv).item()
        return val_loss

    # ── Helper: train reward head ──────────────────────────────────────────
    def train_reward_head(rew_model, Z_rw, R_rw, lr, epochs):
        """Train reward predictor on (latent, true_reward) pairs."""
        if len(Z_rw) == 0:
            print("    ⚠ No reward labels — skipping reward head training")
            return float("nan")
        # Normalize rewards
        r_mean, r_std = R_rw.mean().item(), R_rw.std().item() + 1e-6
        R_norm = (R_rw - r_mean) / r_std
        ds  = TensorDataset(Z_rw, R_norm)
        dl  = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(rew_model.parameters(), lr=lr, weight_decay=1e-4)
        rew_model.train()
        for ep in range(epochs):
            for zb, rb in dl:
                zb, rb = zb.to(device), rb.to(device)
                opt.zero_grad()
                r_hat = rew_model(zb)
                loss  = nn.functional.mse_loss(r_hat, rb)
                loss.backward()
                opt.step()
        n_val = max(1, len(Z_rw) // 10)
        with torch.no_grad():
            zv, rv = Z_rw[-n_val:].to(device), R_norm[-n_val:].to(device)
            val_r  = nn.functional.mse_loss(rew_model(zv), rv).item()
        rew_model.eval()
        return val_r

    # ── [4] Collect initial labelled on-policy data ─────────────────────────
    print("\n[4] Collecting initial labelled rollouts (random policy, with rewards)...")
    env = suite.load("walker", "walk",
                     task_kwargs={"random": 42},
                     environment_kwargs={"flat_observation": False})
    t0 = time.time()
    Z_rw  = []
    A_rw  = []
    Zn_rw = []
    R_rw  = []
    N_SEED = 30
    for i in range(N_SEED):
        ep = collect_episode(env, random_policy, EP_STEPS)
        Z_rw.append(ep["z"]);  A_rw.append(ep["a"])
        Zn_rw.append(ep["z_next"]); R_rw.append(ep["r"])
    Z_rw  = torch.cat(Z_rw,  dim=0)
    A_rw  = torch.cat(A_rw,  dim=0)
    Zn_rw = torch.cat(Zn_rw, dim=0)
    R_rw  = torch.cat(R_rw,  dim=0)
    print(f"    Seed rollouts: {len(Z_rw)} labelled transitions in {(time.time()-t0)/60:.1f} min")

    # Augment offline dataset with labelled data
    Z  = torch.cat([Z,  Z_rw],  dim=0)
    A  = torch.cat([A,  A_rw],  dim=0)
    Zn = torch.cat([Zn, Zn_rw], dim=0)

    # ── [5] Train initial reward head on seed data ─────────────────────────
    print("\n[5] Training initial reward head...")
    t0 = time.time()
    val_rw = train_reward_head(rew_model, Z_rw, R_rw, RW_LR, RW_EPOCHS)
    print(f"    Reward head trained in {(time.time()-t0)/60:.1f} min  val_loss={val_rw:.4f}")

    # ── Helper: evaluate one round ─────────────────────────────────────────
    def evaluate(dyn_model, rew_model, n_eps, ep_steps, seed_offset=0):
        dyn_model.eval(); rew_model.eval()
        mpc_rewards, rand_rewards = [], []
        wins = 0
        for ep_i in range(n_eps):
            env_mpc  = suite.load("walker", "walk",
                                   task_kwargs={"random": seed_offset + ep_i * 2},
                                   environment_kwargs={"flat_observation": False})
            env_rand = suite.load("walker", "walk",
                                   task_kwargs={"random": seed_offset + ep_i * 2 + 1},
                                   environment_kwargs={"flat_observation": False})
            ep_mpc  = collect_episode(env_mpc,  lambda z: cem_policy(z, dyn_model, rew_model), ep_steps)
            ep_rand = collect_episode(env_rand, random_policy, ep_steps)
            r_mpc  = ep_mpc["r"].sum().item()
            r_rand = ep_rand["r"].sum().item()
            win    = r_mpc > r_rand
            wins  += int(win)
            mpc_rewards.append(r_mpc); rand_rewards.append(r_rand)
            mark = "✅" if win else "❌"
            print(f"      ep{ep_i+1}: MPC={r_mpc:.1f}  rand={r_rand:.1f}  {mark}")
        avg_mpc  = np.mean(mpc_rewards)
        avg_rand = np.mean(rand_rewards)
        pct      = wins / n_eps * 100
        return avg_mpc, avg_rand, pct, mpc_rewards, rand_rewards

    # ── [6] Round 0: Baseline with reward head ────────────────────────────
    print("\n--- Round 0: Baseline Eval (Exp6 dynamics + new reward head) ---")
    t0 = time.time()
    r0_mpc, r0_rand, r0_win, r0_mpcs, r0_rands = evaluate(
        dyn_model, rew_model, EVAL_EPS, EP_STEPS)
    print(f"  R0: MPC={r0_mpc:.2f}  rand={r0_rand:.2f}  win={r0_win:.0f}%  [{(time.time()-t0)/60:.1f}min]")

    results = {
        "experiment": 9,
        "env": "walker-walk",
        "approach": "frozen reward head + warm-start dyna",
        "n_dyna_rounds": N_DYNA_ROUNDS,
        "rollouts_per_round": ROLLOUTS_PER_ROUND,
        "ft_lr": FT_LR,
        "ft_epochs": FT_EPOCHS,
        "rw_lr": RW_LR,
        "rw_epochs": RW_EPOCHS,
        "rounds": [{
            "round": 0,
            "n_train": len(Z),
            "avg_mpc_reward": r0_mpc, "avg_rand_reward": r0_rand, "pct_better": r0_win,
            "reward_head_val": val_rw,
        }],
    }

    # ── [7] Dyna rounds ───────────────────────────────────────────────────
    for rnd in range(1, N_DYNA_ROUNDS + 1):
        print(f"\n--- Round {rnd}: Collect {ROLLOUTS_PER_ROUND} on-policy rollouts ---")
        t0 = time.time()
        Z_new, A_new, Zn_new, R_new = [], [], [], []
        for i in range(ROLLOUTS_PER_ROUND):
            seed_i = 1000 + rnd * 1000 + i
            env_i  = suite.load("walker", "walk",
                                  task_kwargs={"random": seed_i},
                                  environment_kwargs={"flat_observation": False})
            ep = collect_episode(env_i, lambda z: cem_policy(z, dyn_model, rew_model), EP_STEPS)
            Z_new.append(ep["z"]);  A_new.append(ep["a"])
            Zn_new.append(ep["z_next"]); R_new.append(ep["r"])

        Z_new  = torch.cat(Z_new,  dim=0)
        A_new  = torch.cat(A_new,  dim=0)
        Zn_new = torch.cat(Zn_new, dim=0)
        R_new  = torch.cat(R_new,  dim=0)

        # Append to cumulative dataset
        Z  = torch.cat([Z,  Z_new],  dim=0)
        A  = torch.cat([A,  A_new],  dim=0)
        Zn = torch.cat([Zn, Zn_new], dim=0)
        Z_rw  = torch.cat([Z_rw, Z_new],  dim=0)
        R_rw  = torch.cat([R_rw, R_new],  dim=0)
        print(f"    Collected {len(Z_new)} on-policy transitions in {(time.time()-t0)/60:.1f} min")
        print(f"    Total dynamics dataset: {len(Z)}")
        print(f"    Total reward dataset:   {len(Z_rw)} labelled")

        # Fine-tune dynamics MLP (warm start)
        print(f"    Fine-tuning dynamics for {FT_EPOCHS} epochs (lr={FT_LR})...")
        t1 = time.time()
        dyn_model.train()
        val_dyn = finetune_dynamics(dyn_model, Z, A, Zn, FT_LR, FT_EPOCHS)
        print(f"    Dynamics FT done in {(time.time()-t1)/60:.1f} min  val_loss={val_dyn:.4f}")

        # Re-train reward head ONLY on R1
        if rnd == 1:
            print(f"    Re-training reward head for {RW_EPOCHS} epochs (lr={RW_LR})...")
            t1 = time.time()
            val_rw = train_reward_head(rew_model, Z_rw, R_rw, RW_LR, RW_EPOCHS)
            print(f"    Reward head done in {(time.time()-t1)/60:.1f} min  val={val_rw:.4f}")
        else:
            print(f"    Keeping reward head frozen from R1 (val={val_rw:.4f}). Skipping re-training.")

        # Evaluate
        print(f"    Evaluating round {rnd}...")
        avg_mpc, avg_rand, pct, _, _ = evaluate(
            dyn_model, rew_model, EVAL_EPS, EP_STEPS, seed_offset=rnd * 100)
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
        torch.save(dyn_model.state_dict(), f"/cache/walker_reward_dyn_r{rnd}.pt")
        torch.save(rew_model.state_dict(), f"/cache/walker_reward_head_r{rnd}.pt")

    # ── [8] Final chart + JSON ─────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds   = [r["round"] for r in results["rounds"]]
    mpc_vals = [r["avg_mpc_reward"] for r in results["rounds"]]
    rnd_vals = [r["avg_rand_reward"] for r in results["rounds"]]
    wins     = [r["pct_better"] for r in results["rounds"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp 9 — Walker-Walk Frozen Reward Head + Dyna Warm-Start",
                 fontsize=13, fontweight="bold")
    ax1.plot(rounds, mpc_vals, "o-", color="#2196F3", lw=2, label="MPC (task reward)")
    ax1.plot(rounds, rnd_vals, "s--", color="#9E9E9E", lw=2, label="Random baseline")
    ax1.set_xlabel("Dyna Round"); ax1.set_ylabel("Avg Episode Reward")
    ax1.set_title("Reward: MPC vs Random"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(rounds, wins, "o-", color="#4CAF50", lw=2, marker="D")
    ax2.axhline(80, color="red", ls="--", alpha=0.5, label="80% target")
    ax2.axhline(50, color="gray", ls=":", alpha=0.4, label="Exp7 plateau")
    ax2.set_xlabel("Dyna Round"); ax2.set_ylabel("Win % (vs random)")
    ax2.set_title("Win Rate"); ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = "/cache/walker_reward_frozen_results.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight"); plt.close()

    # local copy
    chart_local = "/cache/walker_reward_frozen_results.png"

    print(json.dumps(results, indent=2))
    with open("/cache/walker_reward_frozen_results.json", "w") as f:
        json.dumps(results, ensure_ascii=False)
        json.dump(results, f, indent=2)

    print("\n=== EXPERIMENT 9 COMPLETE ===")
    for r in results["rounds"]:
        rw_v = r.get("reward_head_val", float("nan"))
        print(f"  R{r['round']}: MPC={r['avg_mpc_reward']:.2f}  "
              f"rand={r['avg_rand_reward']:.2f}  win={r['pct_better']:.0f}%  "
              f"rw_val={rw_v:.4f}")

    vol_cache.commit()
    print("  ✓ walker_reward_dyna_results.json")
    print("  ✓ walker_reward_dyna_results.png")
    print("  ✓ walker_reward_dyn_rN.pt / walker_reward_head_rN.pt")


@app.local_entrypoint()
def main():
    walker_reward_dyna_experiment.remote()
