"""
V-JEPA 2 — Experiment 3: Iterative Model-Based RL (Dyna Loop)
==============================================================
Hypothesis: The world model improves when trained on data collected
by the MPC policy itself (on-policy / Dyna-style loop) rather than
just on random or goal-conditioned offline data.

Loop structure (3 rounds):
  Round 0 (baseline): Phase 4 dynamics_mlp_ft.pt + MPC eval
  Round 1: Collect K episodes using current MPC policy
            → add to training set → retrain MLP → eval
  Round 2: Collect K more episodes → retrain again → eval

For each round:
  - 10 reacher-easy episodes of MPC policy execution (on-policy rollouts)
  - Mix with existing cached data (goal-directed + random from Phases 2–4)
  - Retrain MLP from scratch on the combined dataset
  - Evaluate: 10 fresh evaluation episodes with new MLP

Metrics per round:
  - Win rate vs random
  - Avg final tip distance
  - Dynamics MLP val_loss

Compute estimate: ~120 min A10G, ~$2.50
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-mbrl")

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
    timeout=14400,
    volumes={
        "/cache":    model_cache,
        "/output":   output_vol,
        "/rollouts": rollout_vol,
    },
)
def iterative_mbrl(
    n_rounds: int = 3,
    collect_episodes_per_round: int = 15,   # MPC policy rollouts per round
    mpc_steps: int = 50,
    horizon: int = 50,                       # use best T from Phase 5
    n_candidates: int = 256,
    n_elites: int = 32,
    n_cem_iters: int = 5,
    ft_epochs: int = 40,
    ft_lr: float = 2e-4,
    batch_size: int = 256,
    n_eval_episodes: int = 10,
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

    # ── Load V-JEPA 2 ────────────────────────────────────────────────────────
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

    # ── MLP definition ───────────────────────────────────────────────────────
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

    def train_mlp(Z_t, A_t, Z_t1, dynamics, label=""):
        N = len(Z_t)
        perm  = np.random.permutation(N)
        split = int(0.85 * N)
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
        val_losses = []

        for epoch in range(ft_epochs):
            dynamics.train()
            for z_b, a_b, z1_b in loader:
                optimizer.zero_grad()
                loss = criterion(dynamics(z_b, a_b), z1_b)
                loss.backward()
                nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            dynamics.eval()
            with torch.no_grad():
                vl = criterion(dynamics(Z_t_te, A_t_te), Z_t1_te).item()
            val_losses.append(vl)
            if (epoch + 1) % 10 == 0:
                print(f"    [{label}] epoch {epoch+1}/{ft_epochs}  val={vl:.4f}")

        return val_losses[-1]

    def run_mpc_collect(dynamics, rng, collect=True):
        """Run MPC policy on reacher-easy, optionally collecting transitions."""
        from dm_control import suite
        env      = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(0, 1000)})
        env_goal = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(1000, 2000)})
        aspec    = env.action_spec()

        frames_out  = []
        actions_out = []

        ts = env.reset()
        env_goal.reset()
        for _ in range(rng.randint(5, 25)):
            env_goal.step(rng.uniform(-1, 1, size=2))
        frame_goal = env_goal.physics.render(height=256, width=256, camera_id=0)
        z_goal     = embed_single(frame_goal).to(DEVICE)
        target_pos = env_goal.physics.named.data.geom_xpos["target", :2].copy()

        for step in range(mpc_steps):
            frame_curr = env.physics.render(height=256, width=256, camera_id=0)
            z_curr     = embed_single(frame_curr).to(DEVICE)
            if collect:
                frames_out.append(frame_curr)

            N_c = n_candidates; K = n_elites; I = n_cem_iters
            mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
            sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
            z_s = z_curr.unsqueeze(0).expand(N_c, -1)
            z_g = z_goal.unsqueeze(0).expand(N_c,  -1)

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

        # Random baseline
        ts = env.reset()
        for _ in range(mpc_steps):
            env.step(rng.uniform(aspec.minimum, aspec.maximum))
        rand_tip  = env.physics.named.data.geom_xpos["finger", :2].copy()
        rand_dist = float(np.linalg.norm(rand_tip - target_pos))

        return {
            "mpc_dist": final_dist, "random_dist": rand_dist,
            "improvement": rand_dist - final_dist,
            "frames": frames_out, "actions": actions_out,
        }

    # ── Load base datasets ────────────────────────────────────────────────
    print("[2] Loading base datasets (goal-directed + random)...")
    Z_t_all, A_t_all, Z_t1_all = [], [], []

    for key in ["reacher_easy_goal", "reacher_easy"]:
        d = Path(f"/rollouts/{key}")
        if not d.exists():
            print(f"    [SKIP] {key} not found in rollout cache")
            continue
        frames  = np.load(str(d / "frames.npy"))
        actions = np.load(str(d / "actions.npy"))
        print(f"    {key}: {len(frames)} transitions")
        z   = embed_batch(frames).numpy()[:-1]
        z1  = embed_batch(frames[1:]).numpy()
        a   = np.array([pad_action(ac) for ac in actions[:-1]], dtype=np.float32)
        Z_t_all.append(z); A_t_all.append(a); Z_t1_all.append(z1)

    Z_t_base  = np.concatenate(Z_t_all,  axis=0)
    A_t_base  = np.concatenate(A_t_all,  axis=0)
    Z_t1_base = np.concatenate(Z_t1_all, axis=0)
    print(f"    Base dataset: {len(Z_t_base)} transitions")

    # ── Iterative loop ────────────────────────────────────────────────────
    print(f"\n[3] Starting MBRL loop ({n_rounds} rounds)...")
    dynamics_curr = DynamicsMLP().to(DEVICE)

    # Load Phase 4 checkpoint as round-0 starting point
    ckpt = torch.load("/output/dynamics_mlp_ft.pt", map_location=DEVICE, weights_only=False)
    dynamics_curr.load_state_dict(ckpt["model_state"])
    print(f"    Loaded Phase 4 checkpoint (val_loss={ckpt['final_val_loss']:.4f}) as round 0")

    Z_t_combined  = Z_t_base.copy()
    A_t_combined  = A_t_base.copy()
    Z_t1_combined = Z_t1_base.copy()

    round_results = []
    rng           = np.random.RandomState(77)

    for round_idx in range(n_rounds):
        print(f"\n  ══ Round {round_idx} ══")

        # --- Evaluate current dynamics ---
        print(f"    Evaluating ({n_eval_episodes} episodes)...")
        eval_res = []
        for ep in range(n_eval_episodes):
            r = run_mpc_collect(dynamics_curr, rng, collect=False)
            eval_res.append(r)
            status = "✅" if r["improvement"] > 0 else "❌"
            print(f"      ep{ep+1}: MPC={r['mpc_dist']:.3f}  rand={r['random_dist']:.3f}  {status}")

        avg_mpc  = float(np.mean([r["mpc_dist"]    for r in eval_res]))
        avg_rand = float(np.mean([r["random_dist"] for r in eval_res]))
        pct      = float(np.mean([r["improvement"] > 0 for r in eval_res])) * 100
        print(f"    Round {round_idx} eval: MPC={avg_mpc:.3f}  rand={avg_rand:.3f}  win={pct:.0f}%")

        round_results.append({
            "round": round_idx,
            "n_train": len(Z_t_combined),
            "avg_mpc_dist": avg_mpc,
            "avg_random_dist": avg_rand,
            "pct_better": pct,
        })

        if round_idx == n_rounds - 1:
            break  # no need to collect/retrain after last eval

        # --- Collect on-policy rollouts with current MPC ---
        print(f"\n    Collecting {collect_episodes_per_round} MPC policy rollouts...")
        new_frames  = []
        new_actions = []
        for ep in range(collect_episodes_per_round):
            r = run_mpc_collect(dynamics_curr, rng, collect=True)
            new_frames.extend(r["frames"])
            new_actions.extend(r["actions"])
            print(f"      Collected ep{ep+1}: {len(r['frames'])} transitions")

        # Embed new transitions
        print(f"    Embedding {len(new_frames) - 1} new transitions...")
        nf_arr = np.array(new_frames, dtype=np.uint8)
        na_arr = np.array(new_actions, dtype=np.float32)
        z_new   = embed_batch(nf_arr).numpy()[:-1]
        z1_new  = embed_batch(nf_arr[1:]).numpy()
        a_new   = na_arr[:-1]

        # Add to combined dataset
        Z_t_combined  = np.concatenate([Z_t_combined,  z_new],  axis=0)
        A_t_combined  = np.concatenate([A_t_combined,  a_new],  axis=0)
        Z_t1_combined = np.concatenate([Z_t1_combined, z1_new], axis=0)
        print(f"    Combined dataset now: {len(Z_t_combined)} transitions")

        # --- Retrain MLP on combined dataset ---
        print(f"    Retraining MLP ({ft_epochs} epochs on {len(Z_t_combined)} transitions)...")
        dynamics_curr = DynamicsMLP().to(DEVICE)  # fresh init
        val_loss = train_mlp(
            Z_t_combined, A_t_combined, Z_t1_combined,
            dynamics_curr, label=f"round{round_idx+1}"
        )
        print(f"    Round {round_idx+1} MLP val_loss: {val_loss:.4f}")
        round_results[-1]["val_loss_after"] = val_loss

        # Save checkpoint
        torch.save({
            "z_dim": z_dim, "max_action_dim": a_pad_dim,
            "hidden_dim": hidden, "n_layers": n_layers,
            "model_state": dynamics_curr.state_dict(),
            "final_val_loss": val_loss, "phase": f"mbrl_r{round_idx+1}",
        }, f"/output/dynamics_mlp_mbrl_r{round_idx+1}.pt")
        output_vol.commit()

    # ── Charts ──────────────────────────────────────────────────────────────
    rounds = [r["round"] for r in round_results]
    mcds   = [r["avg_mpc_dist"]    for r in round_results]
    rcds   = [r["avg_random_dist"] for r in round_results]
    wins   = [r["pct_better"]      for r in round_results]
    Ns     = [r["n_train"]         for r in round_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    ax1.set_facecolor("#1a1a1a"); ax2.set_facecolor("#1a1a1a")

    ax1.plot(rounds, mcds, "o-", color="#4fc3f7", linewidth=2, markersize=8, label="MPC avg dist")
    ax1.plot(rounds, rcds, "s--", color="#ef5350", linewidth=1.5, markersize=6, label="Random avg dist")
    ax1.axhline(0.220, color="#aaa", linestyle=":", linewidth=1, label="Phase 3 (0.220m)")
    ax1.axhline(0.198, color="#ffa726", linestyle=":", linewidth=1, label="Phase 4 (0.198m)")
    ax1.set_xticks(rounds); ax1.set_xlabel("MBRL Round", color="white")
    ax1.set_ylabel("Avg tip distance (m)", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax1.set_title("Tip Distance per MBRL Round", color="white")

    ax2.bar(rounds, wins, color="#66bb6a", width=0.5, edgecolor="white")
    ax2.axhline(60, color="#ffa726", linestyle="--", linewidth=1, label="Phase 3 (60%)")
    ax2.axhline(80, color="#4fc3f7", linestyle="--", linewidth=1, label="Phase 4 (80%)")
    ax2.set_xticks(rounds); ax2.set_xlabel("MBRL Round", color="white")
    ax2.set_ylabel("Win rate vs random (%)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    n_lbls = [f"n={n}" for n in Ns]
    for r, w, lbl in zip(rounds, wins, n_lbls):
        ax2.text(r, w + 1.5, lbl, ha="center", color="white", fontsize=8)
    ax2.set_title("Win Rate per MBRL Round", color="white")

    fig.suptitle(
        f"Experiment 3: Iterative MBRL (Dyna Loop) — reacher-easy\n"
        f"{n_rounds} rounds, {collect_episodes_per_round} rollouts/round, T={horizon}",
        color="white", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("/output/mbrl_results.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── Save JSON ────────────────────────────────────────────────────────────
    final = {
        "experiment": 3,
        "n_rounds": n_rounds,
        "collect_per_round": collect_episodes_per_round,
        "horizon": horizon,
        "rounds": round_results,
    }
    with open("/output/mbrl_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()

    print("\n=== ITERATIVE MBRL SUMMARY ===")
    print(f"  Phase 4 baseline: win=80%  dist=0.198m")
    for r in round_results:
        print(f"  Round {r['round']}: n_train={r['n_train']}  "
              f"MPC={r['avg_mpc_dist']:.3f}m  win={r['pct_better']:.0f}%")
    return final


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Experiment 3: Iterative MBRL")
    print("=" * 60)

    results = iterative_mbrl.remote(
        n_rounds=3,
        collect_episodes_per_round=15,
        mpc_steps=50,
        horizon=50,
        n_candidates=256,
        n_elites=32,
        n_cem_iters=5,
        ft_epochs=40,
        n_eval_episodes=10,
    )

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fname in ["mbrl_results.json", "mbrl_results.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)], check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    print("\nDone!")
