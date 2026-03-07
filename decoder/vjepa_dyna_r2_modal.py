"""
Experiment 4 — Round 2 Continuation
=====================================
The main corrected_dyna script timed out on the Modal heartbeat after successfully
completing Round 1. Round 1 results (confirmed from logs):
  - Round 0: MPC=0.170m  rand=0.184m  win=40%   (Phase 4 FT baseline, this seed)
  - Round 1: MPC=0.151m  rand=0.201m  win=70%   val_loss=0.0144 (warm-start)

This script:
  1. Loads the saved Round 1 checkpoint (dynamics_mlp_dyna_r1.pt)
  2. Collects 50 more MPC rollouts (Round 2 data)
  3. Warm-start fine-tunes from R1 checkpoint (not Phase 4!) with lr=5e-5, 15 epochs
  4. Evaluates Round 2
  5. Writes the final corrected_dyna_results.json + corrected_dyna_results.png
     that covers all 3 rounds (0, 1, 2)
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-dyna-r2")

model_cache = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf", "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scipy tqdm imageio[ffmpeg]",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/cache":    model_cache,
        "/output":   output_vol,
        "/rollouts": rollout_vol,
    },
)
def dyna_round2(
    mpc_steps: int = 50,
    horizon: int = 50,
    n_candidates: int = 256,
    n_elites: int = 32,
    n_cem_iters: int = 5,
    ft_epochs: int = 15,
    ft_lr: float = 5e-5,
    batch_size: int = 256,
    n_eval_episodes: int = 10,
    collect_episodes: int = 50,
):
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    DEVICE = "cuda"

    print("[1] Loading V-JEPA 2...")
    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()

    ET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def embed_single(frame_np):
        img  = Image.fromarray(frame_np)
        clip = ET(img).unsqueeze(0).repeat(8, 1, 1, 1).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=clip)
            return out.last_hidden_state.mean(dim=1).squeeze(0).float()

    def embed_batch(frames_np, bs=32):
        embs = []
        for s in range(0, len(frames_np), bs):
            batch = frames_np[s:s + bs]
            clips = []
            for f in batch:
                img  = Image.fromarray(f)
                clip = ET(img).unsqueeze(0).repeat(8, 1, 1, 1)
                clips.append(clip)
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(embs, dim=0)

    z_dim     = 1024
    a_pad_dim = 6
    hidden    = 512
    n_layers  = 3

    class DynamicsMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []; in_d = z_dim + a_pad_dim
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU()]
                in_d = hidden
            layers.append(nn.Linear(in_d, z_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1))

    def pad_action(a_np, pad_to=6):
        return np.concatenate([a_np, np.zeros(pad_to - len(a_np), dtype=np.float32)])

    def warm_finetune(Z_t, A_t, Z_t1, dynamics, label=""):
        perm   = np.random.permutation(len(Z_t))
        split  = int(0.85 * len(Z_t))
        tr, te = perm[:split], perm[split:]
        Z_t_tr  = torch.tensor(Z_t[tr]).to(DEVICE)
        A_t_tr  = torch.tensor(A_t[tr]).to(DEVICE)
        Z_t1_tr = torch.tensor(Z_t1[tr]).to(DEVICE)
        Z_t_te  = torch.tensor(Z_t[te]).to(DEVICE)
        A_t_te  = torch.tensor(A_t[te]).to(DEVICE)
        Z_t1_te = torch.tensor(Z_t1[te]).to(DEVICE)
        dataset   = TensorDataset(Z_t_tr, A_t_tr, Z_t1_tr)
        loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=ft_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
        criterion = nn.MSELoss()
        dynamics.train()
        for epoch in range(ft_epochs):
            for z_b, a_b, z1_b in loader:
                optimizer.zero_grad()
                loss = criterion(dynamics(z_b, a_b), z1_b)
                loss.backward()
                nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                dynamics.eval()
                with torch.no_grad():
                    vl = criterion(dynamics(Z_t_te, A_t_te), Z_t1_te).item()
                dynamics.train()
                print(f"  [{label}] epoch {epoch+1}/{ft_epochs}  val={vl:.4f}")
        dynamics.eval()
        with torch.no_grad():
            final_vl = criterion(dynamics(Z_t_te, A_t_te), Z_t1_te).item()
        return final_vl

    def run_mpc_episode(dynamics, rng, collect=True):
        from dm_control import suite
        env      = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(0, 1000)})
        env_goal = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(1000, 2000)})
        aspec    = env.action_spec()
        frames_out = []; actions_out = []
        env.reset(); env_goal.reset()
        for _ in range(rng.randint(5, 25)):
            env_goal.step(rng.uniform(-1, 1, size=2))
        frame_goal = env_goal.physics.render(height=256, width=256, camera_id=0)
        z_goal     = embed_single(frame_goal).to(DEVICE)
        target_pos = env_goal.physics.named.data.geom_xpos["target", :2].copy()
        for step in range(mpc_steps):
            frame_curr = env.physics.render(height=256, width=256, camera_id=0)
            z_curr = embed_single(frame_curr).to(DEVICE)
            if collect:
                frames_out.append(frame_curr)
            N_c = n_candidates; K = n_elites; I = n_cem_iters
            mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
            sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
            z_s = z_curr.unsqueeze(0).expand(N_c, -1)
            z_g = z_goal.unsqueeze(0).expand(N_c, -1)
            for _ in range(I):
                eps      = rng.randn(N_c, horizon, a_pad_dim).astype(np.float32)
                act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
                act_t    = torch.tensor(act_seqs, device=DEVICE)
                z_c      = z_s.clone()
                with torch.no_grad():
                    for t in range(horizon):
                        z_c = dynamics(z_c, act_t[:, t, :])
                costs     = ((z_c - z_g) ** 2).sum(dim=-1).cpu().numpy()
                elite_idx = np.argsort(costs)[:K]
                mu  = act_seqs[elite_idx].mean(axis=0)
                sig = act_seqs[elite_idx].std(axis=0) + 1e-6
            a_exec = np.clip(mu[0, :2], aspec.minimum, aspec.maximum)
            if collect:
                actions_out.append(pad_action(a_exec))
            env.step(a_exec)
        final_tip  = env.physics.named.data.geom_xpos["finger", :2].copy()
        final_dist = float(np.linalg.norm(final_tip - target_pos))
        env.reset()
        for _ in range(mpc_steps):
            env.step(rng.uniform(aspec.minimum, aspec.maximum))
        rand_tip  = env.physics.named.data.geom_xpos["finger", :2].copy()
        rand_dist = float(np.linalg.norm(rand_tip - target_pos))
        return {
            "mpc_dist": final_dist, "random_dist": rand_dist,
            "improvement": rand_dist - final_dist,
            "frames": frames_out, "actions": actions_out,
        }

    # --- Load R1 combined data (base + R0 rollouts) ---
    print("[2] Loading base datasets...")
    Z_t_all, A_t_all, Z_t1_all = [], [], []
    for key in ["reacher_easy_goal", "reacher_easy"]:
        d = Path(f"/rollouts/{key}")
        if not d.exists():
            print(f"  [SKIP] {key}")
            continue
        frames  = np.load(str(d / "frames.npy"))
        actions = np.load(str(d / "actions.npy"))
        print(f"  {key}: {len(frames)} transitions")
        z   = embed_batch(frames).numpy()[:-1]
        z1  = embed_batch(frames[1:]).numpy()
        a   = np.array([pad_action(ac) for ac in actions[:-1]], dtype=np.float32)
        Z_t_all.append(z); A_t_all.append(a); Z_t1_all.append(z1)
    Z_t_combined  = np.concatenate(Z_t_all,  axis=0)
    A_t_combined  = np.concatenate(A_t_all,  axis=0)
    Z_t1_combined = np.concatenate(Z_t1_all, axis=0)
    print(f"  Base: {len(Z_t_combined)} transitions")

    print("[3] Loading R1 checkpoint (warm start for Round 2)...")
    # Find the R1 checkpoint file
    output_path = Path("/output")
    r1_files = sorted(output_path.glob("dynamics_mlp_dyna_r1*.pt"))
    if not r1_files:
        raise FileNotFoundError("No r1 checkpoint found! Cannot continue.")
    r1_ckpt_path = r1_files[-1]
    print(f"  Using: {r1_ckpt_path.name}")
    r1_ckpt = torch.load(str(r1_ckpt_path), map_location=DEVICE, weights_only=False)
    r1_state = {k: v.clone() for k, v in r1_ckpt["model_state"].items()}
    print(f"  R1 val_loss={r1_ckpt['final_val_loss']:.4f}")

    rng = np.random.RandomState(99 + 100)  # offset seed to avoid repeating R0/R1 data
    dynamics = DynamicsMLP().to(DEVICE)
    dynamics.load_state_dict(r1_state)

    print(f"\n[4] Collecting {collect_episodes} MPC rollouts for Round 2...")
    new_frames, new_actions = [], []
    for ep in range(collect_episodes):
        r = run_mpc_episode(dynamics, rng, collect=True)
        new_frames.extend(r["frames"]); new_actions.extend(r["actions"])
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}/{collect_episodes}")
    print(f"  Collected {len(new_frames)} transitions. Embedding...")
    nf_arr = np.array(new_frames, dtype=np.uint8)
    na_arr = np.array(new_actions, dtype=np.float32)
    z_new  = embed_batch(nf_arr).numpy()[:-1]
    z1_new = embed_batch(nf_arr[1:]).numpy()
    a_new  = na_arr[:-1]
    Z_t_combined  = np.concatenate([Z_t_combined,  z_new],  axis=0)
    A_t_combined  = np.concatenate([A_t_combined,  a_new],  axis=0)
    Z_t1_combined = np.concatenate([Z_t1_combined, z1_new], axis=0)
    print(f"  Combined (R2): {len(Z_t_combined)} transitions")

    print(f"\n[5] Warm-start fine-tune R2 ({ft_epochs} epochs, lr={ft_lr})...")
    dynamics_r2 = DynamicsMLP().to(DEVICE)
    dynamics_r2.load_state_dict(r1_state)  # warm-start from R1 not Phase 4
    val_r2 = warm_finetune(Z_t_combined, A_t_combined, Z_t1_combined, dynamics_r2, "r2")
    print(f"  R2 val_loss={val_r2:.4f}")
    torch.save({
        "z_dim": z_dim, "max_action_dim": a_pad_dim,
        "hidden_dim": hidden, "n_layers": n_layers,
        "model_state": dynamics_r2.state_dict(),
        "final_val_loss": val_r2,
        "phase": "corrected_dyna_r2",
    }, "/output/dynamics_mlp_dyna_r2.pt")
    output_vol.commit()

    print(f"\n[6] Evaluating Round 2 ({n_eval_episodes} episodes)...")
    rng2 = np.random.RandomState(42)
    eval_res = []
    for ep in range(n_eval_episodes):
        r = run_mpc_episode(dynamics_r2, rng2, collect=False)
        eval_res.append(r)
        status = "✅" if r["improvement"] > 0 else "❌"
        print(f"  ep{ep+1}: MPC={r['mpc_dist']:.3f}  rand={r['random_dist']:.3f}  {status}")
    avg_mpc_r2  = float(np.mean([r["mpc_dist"]    for r in eval_res]))
    avg_rand_r2 = float(np.mean([r["random_dist"] for r in eval_res]))
    win_r2      = float(np.mean([r["improvement"] > 0 for r in eval_res])) * 100
    print(f"  Round 2: MPC={avg_mpc_r2:.3f}  rand={avg_rand_r2:.3f}  win={win_r2:.0f}%")

    # Combine with known R0/R1 results (from logs)
    all_rounds = [
        {"round": 0, "n_train": 14923, "avg_mpc_dist": 0.170, "avg_random_dist": 0.184, "pct_better": 40.0},
        {"round": 1, "n_train": 17422, "avg_mpc_dist": 0.151, "avg_random_dist": 0.201, "pct_better": 70.0, "val_loss_after": 0.0144},
        {"round": 2, "n_train": len(Z_t_combined), "avg_mpc_dist": avg_mpc_r2, "avg_random_dist": avg_rand_r2, "pct_better": win_r2, "val_loss_after": val_r2},
    ]

    # Chart
    rounds = [r["round"] for r in all_rounds]
    mcds   = [r["avg_mpc_dist"] for r in all_rounds]
    rdists = [r["avg_random_dist"] for r in all_rounds]
    wins   = [r["pct_better"] for r in all_rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    ax1.set_facecolor("#1a1a1a"); ax2.set_facecolor("#1a1a1a")

    ax1.plot(rounds, mcds,   "o-",  color="#4fc3f7", linewidth=2, markersize=9, label="MPC (corrected Dyna)")
    ax1.plot(rounds, rdists, "s--", color="#ef5350", linewidth=1.5, markersize=7, label="Random baseline")
    ax1.axhline(0.198, color="#ffa726", linestyle=":", linewidth=1.5, label="Phase 4 FT (0.198m)")
    ax1.axhline(0.226, color="#e91e63", linestyle=":", linewidth=1, label="Exp3 R1 best (0.226m)")
    ax1.set_xlabel("Round", color="white"); ax1.set_ylabel("Avg tip distance (m)", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax1.set_title("Corrected Dyna: Tip Distance per Round", color="white")
    ax1.set_xticks(rounds)
    for r, d in zip(rounds, mcds):
        ax1.text(r, d + 0.003, f"{d:.3f}m", ha="center", color="#4fc3f7", fontsize=8.5)

    ax2.bar(rounds, wins, color=["#ef9a9a", "#66bb6a", "#a5d6a7"], width=0.5, edgecolor="white")
    ax2.axhline(70, color="#ffa726", linestyle="--", linewidth=1.5, label="Exp3 R0 baseline (70%)")
    ax2.axhline(80, color="#4fc3f7", linestyle="--", linewidth=1.5, label="Phase 4 FT (80%)")
    for r, w in zip(rounds, wins):
        ax2.text(r, w + 1.5, f"{w:.0f}%", ha="center", color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round", color="white"); ax2.set_ylabel("Win rate (%)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax2.set_title("Corrected Dyna: Win Rate per Round", color="white")
    ax2.set_xticks(rounds)

    phase4_str = "Phase 4 FT: 80% win, 0.198m" if wins[-1] < 80 else f"Round 2: {wins[-1]:.0f}% win ✅"
    fig.suptitle(
        f"Experiment 4: Corrected Dyna (Warm Start) — reacher-easy\n"
        f"3 rounds × 50 rollouts | T=50 | warm-start from Phase 4 FT | lr={ft_lr}",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("/output/corrected_dyna_results.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    final = {
        "experiment": 4,
        "n_rounds": 3,
        "collect_per_round": collect_episodes,
        "ft_lr": ft_lr, "ft_epochs": ft_epochs,
        "warmstart": True,
        "r1_checkpoint": r1_ckpt_path.name,
        "rounds": all_rounds,
    }
    with open("/output/corrected_dyna_results.json", "w") as f:
        import json; json.dump(final, f, indent=2)
    output_vol.commit()

    print("\n=== EXPERIMENT 4 COMPLETE ===")
    print(f"  Round 0: 40% win  0.170m  (baseline this seed)")
    print(f"  Round 1: 70% win  0.151m  val={0.0144:.4f}  (warm-start from Phase4 FT)")
    print(f"  Round 2: {win_r2:.0f}% win  {avg_mpc_r2:.3f}m  val={val_r2:.4f}  (warm-start from R1)")
    return final


@app.local_entrypoint()
def main():
    import subprocess
    from pathlib import Path

    print("=" * 60)
    print("Experiment 4 — Round 2 continuation (warm-start)")
    print("=" * 60)
    results = dyna_round2.remote()

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fname in ["corrected_dyna_results.json", "corrected_dyna_results.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)], check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    print("Done! Check decoder_output/corrected_dyna_results.{json,png}")
