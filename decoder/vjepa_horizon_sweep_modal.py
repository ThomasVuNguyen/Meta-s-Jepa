"""
V-JEPA 2 — Phase 5: Horizon Sweep + Reacher-Hard Transfer
===========================================================
Two sub-experiments in one job:

  5a. Horizon sweep on reacher-easy
        Use Phase 4 fine-tuned MLP (dynamics_mlp_ft.pt) and sweep
        planning horizon T ∈ {25, 50, 75} with replan-every-step MPC.
        Shows the saturation point where longer horizons stop helping.

  5b. Transfer to reacher-hard
        Same MPC pipeline (T=50, best horizon from 5a) applied to
        reacher-hard — a harder variant where the target zone is smaller.
        Does goal-conditioned fine-tuning generalise?

Hypothesis:
  - T=50 should outperform T=25 (Phase 4) by handling larger workspace gaps
  - T=75 may show diminishing returns or degrade (compounding MLP errors)
  - reacher-hard win rate will be lower than easy but still above 60%

Compute estimate: ~45 min A10G, ~$1.00
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-horizon-sweep")

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
def horizon_sweep_and_transfer(
    horizons: list = [25, 50, 75],
    n_episodes: int = 10,
    mpc_steps: int = 60,
    n_candidates: int = 256,
    n_elites: int = 32,
    n_cem_iters: int = 5,
):
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
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

    # ── Load fine-tuned dynamics MLP ─────────────────────────────────────────
    print("[2] Loading Phase 4 fine-tuned dynamics MLP...")
    ckpt = torch.load("/output/dynamics_mlp_ft.pt", map_location=DEVICE,
                      weights_only=False)
    z_dim     = ckpt["z_dim"]          # 1024
    a_pad_dim = ckpt["max_action_dim"] # 6

    class DynamicsMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
            super().__init__()
            layers = []; in_d = input_dim
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, output_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1))

    dynamics = DynamicsMLP(z_dim + a_pad_dim, ckpt["hidden_dim"], z_dim, ckpt["n_layers"]).to(DEVICE)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()
    print(f"    Phase 4 val_loss={ckpt['final_val_loss']:.4f}")

    def pad_action(a_np, pad_to=6):
        pad = np.zeros(pad_to - len(a_np), dtype=np.float32)
        return np.concatenate([a_np, pad])

    # ── MPC runner ─────────────────────────────────────────────────────────
    def run_mpc_episodes(env_domain, env_task, horizon, label):
        from dm_control import suite

        env      = suite.load(env_domain, env_task, task_kwargs={"random": 0})
        env_goal = suite.load(env_domain, env_task, task_kwargs={"random": 1})
        aspec    = env.action_spec()
        rng      = np.random.RandomState(42 + horizon)
        ep_results = []

        for ep_idx in range(n_episodes):
            ts = env.reset()

            # Sample random goal configuration
            env_goal.reset()
            for _ in range(rng.randint(5, 25)):
                env_goal.step(rng.uniform(-1, 1, size=2))
            frame_goal = env_goal.physics.render(height=256, width=256, camera_id=0)
            z_goal     = embed_single(frame_goal).to(DEVICE)
            target_pos = env_goal.physics.named.data.geom_xpos["target", :2].copy()

            last_action = None
            for step in range(mpc_steps):
                frame_curr = env.physics.render(height=256, width=256, camera_id=0)
                z_curr     = embed_single(frame_curr).to(DEVICE)

                # CEM
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

                # Execute first action
                a_exec = np.clip(mu[0, :2], aspec.minimum, aspec.maximum)
                env.step(a_exec)

            final_tip  = env.physics.named.data.geom_xpos["finger", :2].copy()
            final_dist = float(np.linalg.norm(final_tip - target_pos))

            # Random baseline
            ts = env.reset()
            for _ in range(mpc_steps):
                env.step(rng.uniform(aspec.minimum, aspec.maximum))
            rand_tip  = env.physics.named.data.geom_xpos["finger", :2].copy()
            rand_dist = float(np.linalg.norm(rand_tip - target_pos))

            ep_results.append({
                "episode": ep_idx, "mpc_dist": final_dist, "random_dist": rand_dist,
                "improvement": rand_dist - final_dist,
            })
            print(f"    [{label} T={horizon}] ep{ep_idx+1}: MPC={final_dist:.3f}  rand={rand_dist:.3f}  "
                  f"{'✅' if rand_dist > final_dist else '❌'}")

        return ep_results

    # ── 5a: Horizon sweep on reacher-easy ─────────────────────────────────
    print(f"\n[3] EXPERIMENT 5a — Horizon sweep {horizons} on reacher-easy")
    sweep_results = {}
    for T in horizons:
        print(f"\n  Horizon T={T}...")
        results = run_mpc_episodes("reacher", "easy", horizon=T, label="easy")
        avg_mpc  = float(np.mean([r["mpc_dist"]    for r in results]))
        avg_rand = float(np.mean([r["random_dist"] for r in results]))
        pct      = float(np.mean([r["improvement"] > 0 for r in results])) * 100
        sweep_results[T] = {
            "horizon": T, "avg_mpc_dist": avg_mpc, "avg_random_dist": avg_rand,
            "pct_better": pct, "episodes": results,
        }
        print(f"  → Avg MPC={avg_mpc:.3f}m  random={avg_rand:.3f}m  win={pct:.0f}%")

    # ── 5b: Transfer to reacher-hard ──────────────────────────────────────
    best_T = max(sweep_results, key=lambda t: sweep_results[t]["pct_better"])
    print(f"\n[4] EXPERIMENT 5b — Transfer to reacher-hard (best T={best_T})")
    hard_results = run_mpc_episodes("reacher", "hard", horizon=best_T, label="hard")
    avg_mpc_hard  = float(np.mean([r["mpc_dist"]    for r in hard_results]))
    avg_rand_hard = float(np.mean([r["random_dist"] for r in hard_results]))
    pct_hard      = float(np.mean([r["improvement"] > 0 for r in hard_results])) * 100
    print(f"  → Hard: MPC={avg_mpc_hard:.3f}m  random={avg_rand_hard:.3f}m  win={pct_hard:.0f}%")

    # ── Charts ─────────────────────────────────────────────────────────────
    # 1. Horizon sweep: win rate + avg dist
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    ax1.set_facecolor("#1a1a1a"); ax2.set_facecolor("#1a1a1a")

    Ts         = list(sweep_results.keys())
    win_rates  = [sweep_results[t]["pct_better"]  for t in Ts]
    avg_dists  = [sweep_results[t]["avg_mpc_dist"] for t in Ts]
    rand_dists = [sweep_results[0 if 0 in sweep_results else Ts[0]]["avg_random_dist"]] * len(Ts)

    ax1.plot(Ts, win_rates, "o-", color="#4fc3f7", linewidth=2, markersize=8)
    ax1.axhline(60, color="#ffa726", linestyle="--", linewidth=1, label="Phase 3 (CEM, T=10)")
    ax1.axhline(80, color="#66bb6a", linestyle="--", linewidth=1, label="Phase 4 (MPC, T=25)")
    ax1.set_xlabel("Horizon T", color="white"); ax1.set_ylabel("Win rate (%)", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax1.set_title("Win Rate vs Planning Horizon", color="white")

    ax2.plot(Ts, avg_dists, "o-", color="#4fc3f7", linewidth=2, markersize=8, label="MPC avg dist")
    ax2.axhline(0.220, color="#ffa726", linestyle="--", linewidth=1, label="Phase 3 (0.220m)")
    ax2.axhline(0.198, color="#66bb6a", linestyle="--", linewidth=1, label="Phase 4 (0.198m)")
    ax2.set_xlabel("Horizon T", color="white"); ax2.set_ylabel("Avg tip dist (m)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax2.set_title("Tip Distance vs Planning Horizon", color="white")

    [ax.set_xticks(Ts) for ax in (ax1, ax2)]
    fig.suptitle("Phase 5a — Horizon Sweep on reacher-easy", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("/output/horizon_sweep.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # 2. Easy vs Hard comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    labels = ["Phase 3\nCEM (easy)", "Phase 4\nMPC (easy)", f"Phase 5a\nMPC T={best_T} (easy)",
              f"Phase 5b\nMPC T={best_T} (hard)", "Random (easy)", "Random (hard)"]
    values = [0.220, 0.198,
              sweep_results[best_T]["avg_mpc_dist"],
              avg_mpc_hard,
              sweep_results[best_T]["avg_random_dist"],
              avg_rand_hard]
    colors = ["#ffa726", "#66bb6a", "#4fc3f7", "#ab47bc", "#ef5350", "#e57373"]
    bars   = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}m",
                ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg fingertip distance to goal (m)", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.set_title(f"Phase 5 Summary: Easy vs Hard Transfer\n"
                 f"Best horizon T={best_T}, win rates: easy={sweep_results[best_T]['pct_better']:.0f}% "
                 f"hard={pct_hard:.0f}%", color="white")
    plt.tight_layout()
    plt.savefig("/output/horizon_easy_vs_hard.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── Save JSON ──────────────────────────────────────────────────────────
    final = {
        "phase": "5a+5b",
        "sweep": sweep_results,
        "best_T": best_T,
        "hard_transfer": {
            "avg_mpc_dist": avg_mpc_hard,
            "avg_random_dist": avg_rand_hard,
            "pct_better": pct_hard,
            "episodes": hard_results,
        },
    }
    with open("/output/horizon_sweep_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()
    return final


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Phase 5: Horizon Sweep + Reacher-Hard")
    print("=" * 60)

    results = horizon_sweep_and_transfer.remote(
        horizons=[25, 50, 75],
        n_episodes=10,
        mpc_steps=60,
        n_candidates=256,
        n_elites=32,
        n_cem_iters=5,
    )

    print("\n=== FINAL SUMMARY ===")
    for T, r in results["sweep"].items():
        print(f"  T={T}: MPC={r['avg_mpc_dist']:.3f}m  win={r['pct_better']:.0f}%")
    print(f"  Best T={results['best_T']}")
    print(f"  Hard transfer: MPC={results['hard_transfer']['avg_mpc_dist']:.3f}m  "
          f"win={results['hard_transfer']['pct_better']:.0f}%")

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fname in ["horizon_sweep_results.json", "horizon_sweep.png", "horizon_easy_vs_hard.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)], check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    print("\nDone!")
