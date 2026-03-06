"""
V-JEPA 2 CEM Goal-Reaching Planner — Phase 3 of Meta-s-Jepa
=============================================================
Uses the trained dynamics MLP to plan action sequences via
Cross-Entropy Method (CEM) that steer an agent toward a goal
frame, entirely in V-JEPA 2 latent space. No reward function,
no task-specific training — just frozen embeddings + dynamics.

Pipeline:
  1. Load trained dynamics MLP from Phase 2 (dynamics_mlp.pt)
  2. Spawn reacher-easy episodes:
       - Sample a random start state
       - Sample a random goal arm pose → render goal frame → z_goal
  3. CEM planning loop (pure latent space):
       repeat for n_iter:
         - Sample N candidate action sequences of length T
         - Roll out forward: z_1 = f(z_0, a_0), z_2 = f(z_1, a_1), ...
         - Compute cost: ||z_T - z_goal||²
         - Keep top K elite sequences
         - Refit Gaussian μ, σ from elites
       → Execute best action sequence in environment
  4. Evaluate:
       - Euclidean tip-to-target distance (from dm_control state)
       - Compare: CEM planner vs random baseline vs zero-action baseline
  5. Record a video of the planned episode
  6. Save results to vjepa2-decoder-output

Why CEM?
  - CEM is the standard first choice for latent-space MBRL
    (used in DreamerV1, Dreaming paper, TDMPC)
  - No gradient needed — works with any black-box dynamics model
  - Parallelisable: all N trajectories can be batched on GPU

Compute estimate:
  - No training — pure inference on CPU/GPU
  - N=512 candidates, T=10 horizon, n_iter=10 → 51,200 MLP forward passes
  - MLP is tiny (1.3M params) → all 512 rollouts fit in a single GPU batch
  - Expected runtime: ~5 min on A10G (mostly env rendering + embedding)
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-cem-planner")

model_cache = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf",
        "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scipy tqdm imageio[ffmpeg]",
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# CEM Planner
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        "/cache":  model_cache,
        "/output": output_vol,
    },
)
def run_cem_planner(
    n_episodes: int = 10,
    # CEM hyperparameters
    horizon: int = 10,       # planning horizon T (steps)
    n_candidates: int = 512, # candidate sequences per CEM iteration
    n_elites: int = 64,      # top-K elites kept per iteration
    n_cem_iters: int = 10,   # CEM refinement iterations
    action_dim: int = 2,     # reacher-easy has action_dim=2
    # MLP architecture (must match Phase 2)
    hidden_dim: int = 512,
    n_layers: int = 3,
    max_action_dim: int = 6,
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
    import imageio

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    DEVICE = "cuda"

    # ── 1. Load V-JEPA 2 ─────────────────────────────────────────────────────
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

    def embed_frame(frame_np):
        """Single frame (H,W,3) uint8 → (1024,) tensor on DEVICE."""
        img  = Image.fromarray(frame_np)
        clip = ET(img).unsqueeze(0).repeat(8, 1, 1, 1).unsqueeze(0)  # [1,8,C,H,W]
        clip = clip.to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=clip)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).float()  # (1024,)
        return emb

    def embed_batch(frames_np, bs=32):
        """(N, H, W, 3) uint8 → (N, 1024) float32 tensor on CPU."""
        embs = []
        for start in range(0, len(frames_np), bs):
            batch = frames_np[start:start + bs]
            clips = []
            for f in batch:
                img  = Image.fromarray(f)
                clip = ET(img).unsqueeze(0).repeat(8, 1, 1, 1)
                clips.append(clip)
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out  = vjepa(pixel_values_videos=clips)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(embs, dim=0)

    # ── 2. Load dynamics MLP ─────────────────────────────────────────────────
    print("[2] Loading dynamics MLP from Phase 2...")
    ckpt_path = Path("/output/dynamics_mlp.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "dynamics_mlp.pt not found in /output/. Run vjepa_dynamics_modal.py first."
        )
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)

    z_dim     = ckpt["z_dim"]         # 1024
    a_pad_dim = ckpt["max_action_dim"] # 6 (padded)
    input_dim = z_dim + a_pad_dim      # 1030

    class DynamicsMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
            super().__init__()
            layers = []
            in_d = input_dim
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1))

    dynamics = DynamicsMLP(input_dim, ckpt["hidden_dim"], z_dim, ckpt["n_layers"]).to(DEVICE)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()
    print(f"    Loaded: {ckpt['n_params']:,} params, final_val_loss={ckpt['final_val_loss']:.4f}")

    def pad_action(a_np):
        """Pad action from action_dim to max_action_dim and return as tensor."""
        pad = np.zeros(a_pad_dim - len(a_np), dtype=np.float32)
        return torch.tensor(np.concatenate([a_np, pad]), device=DEVICE).float()

    # ── 3. CEM Planning Function ─────────────────────────────────────────────
    def cem_plan(z_start, z_goal, action_dim, rng):
        """
        CEM loop: find action sequence [a_0, ..., a_{T-1}] that minimises
            cost = ||f^T(z_start, a) - z_goal||²
        Returns (best_actions, best_cost, cost_history)
        """
        T    = horizon
        N    = n_candidates
        K    = n_elites
        I    = n_cem_iters

        # Initial action distribution: μ=0, σ=1 for all (T, a_pad_dim)
        mu  = np.zeros((T, a_pad_dim), dtype=np.float32)
        sig = np.ones((T,  a_pad_dim), dtype=np.float32)

        z_s = z_start.unsqueeze(0).expand(N, -1)  # (N, 1024)
        z_g = z_goal.unsqueeze(0).expand(N, -1)   # (N, 1024)

        cost_history = []
        best_actions = None
        best_cost    = float("inf")

        for it in range(I):
            # Sample N candidate action sequences: (N, T, a_pad_dim)
            eps      = rng.randn(N, T, a_pad_dim).astype(np.float32)
            act_seqs = mu[None] + sig[None] * eps  # (N, T, a_pad_dim)
            # Clip to valid action range [-1, 1]
            act_seqs = np.clip(act_seqs, -1.0, 1.0)
            act_t    = torch.tensor(act_seqs, device=DEVICE)  # (N, T, a_pad_dim)

            # Roll out N sequences in parallel through dynamics MLP
            z_curr = z_s.clone()  # (N, 1024)
            with torch.no_grad():
                for t in range(T):
                    a_t    = act_t[:, t, :]          # (N, a_pad_dim)
                    z_curr = dynamics(z_curr, a_t)   # (N, 1024)

            # Compute latent distance cost: ||z_T - z_goal||²
            costs = ((z_curr - z_g) ** 2).sum(dim=-1).cpu().numpy()  # (N,)

            # Select elites
            elite_idx  = np.argsort(costs)[:K]
            elite_seqs = act_seqs[elite_idx]  # (K, T, a_pad_dim)
            elite_costs = costs[elite_idx]

            # Refit Gaussian from elites
            mu  = elite_seqs.mean(axis=0)
            sig = elite_seqs.std(axis=0) + 1e-6

            iter_best = float(elite_costs[0])
            cost_history.append(iter_best)

            if iter_best < best_cost:
                best_cost    = iter_best
                best_actions = act_seqs[elite_idx[0]]  # (T, a_pad_dim)

        return best_actions, best_cost, cost_history

    # ── 4. Run Episodes ───────────────────────────────────────────────────────
    print(f"\n[3] Running {n_episodes} reacher-easy episodes with CEM planner...")
    from dm_control import suite as dmc_suite

    env = dmc_suite.load("reacher", "easy", task_kwargs={"random": 0})
    env_rand_baseline = dmc_suite.load("reacher", "easy", task_kwargs={"random": 1})
    action_spec = env.action_spec()

    rng = np.random.RandomState(42)

    ep_results = []
    all_cost_curves = []

    for ep_idx in range(n_episodes):
        print(f"\n  Episode {ep_idx + 1}/{n_episodes}")
        # Reset env
        ts = env.reset()

        # Render start frame, get z_start
        frame_start = env.physics.render(height=256, width=256, camera_id=0)
        z_start = embed_frame(frame_start).to(DEVICE)

        # Sample a goal: advance environment with random actions for 10 steps
        # to get a "goal" state, render it, embed it
        ts_goal = env_rand_baseline.reset()
        for _ in range(rng.randint(5, 20)):
            rand_a = rng.uniform(action_spec.minimum, action_spec.maximum)
            ts_goal = env_rand_baseline.step(rand_a)
        frame_goal = env_rand_baseline.physics.render(height=256, width=256, camera_id=0)
        z_goal = embed_frame(frame_goal).to(DEVICE)

        # True goal position from physics (fingertip distance to target)
        goal_target_pos = env_rand_baseline.physics.named.data.geom_xpos["target", :2]

        # CEM plan
        print(f"    Planning with CEM (horizon={horizon}, N={n_candidates}, iters={n_cem_iters})...")
        best_actions, plan_cost, cost_history = cem_plan(z_start, z_goal, action_dim, rng)
        print(f"    CEM final cost: {plan_cost:.4f}  (start: {cost_history[0]:.4f})")
        all_cost_curves.append(cost_history)

        # Execute planned action sequence in environment
        frames_planned = [frame_start]
        ts = env.reset()
        for t in range(horizon):
            a_full   = best_actions[t, :action_dim]  # clip back to actual action_dim
            a_clipped = np.clip(a_full, action_spec.minimum, action_spec.maximum)
            ts = env.step(a_clipped)
            frame = env.physics.render(height=256, width=256, camera_id=0)
            frames_planned.append(frame)
        planned_tip_pos = env.physics.named.data.geom_xpos["finger", :2]
        planned_dist    = float(np.linalg.norm(planned_tip_pos - goal_target_pos))

        # Random baseline: execute T random actions
        ts = env.reset()
        for _ in range(horizon):
            rand_a = rng.uniform(action_spec.minimum, action_spec.maximum)
            ts     = env.step(rand_a)
        random_tip_pos = env.physics.named.data.geom_xpos["finger", :2]
        random_dist    = float(np.linalg.norm(random_tip_pos - goal_target_pos))

        ep_result = {
            "episode":       ep_idx,
            "plan_cost_final": plan_cost,
            "plan_cost_start": cost_history[0],
            "planned_tip_dist_to_goal": planned_dist,
            "random_tip_dist_to_goal":  random_dist,
            "improvement_over_random":  random_dist - planned_dist,
        }
        ep_results.append(ep_result)
        print(f"    Tip dist to goal — CEM: {planned_dist:.4f}  Random: {random_dist:.4f}  "
              f"Δ={random_dist - planned_dist:+.4f}")

        # Save planned episode video (first 3 episodes)
        if ep_idx < 3:
            vid_path = f"/output/cem_episode_{ep_idx}.mp4"
            writer   = imageio.get_writer(vid_path, fps=10)
            for fr in frames_planned:
                writer.append_data(fr)
            writer.close()
            print(f"    Saved video: {vid_path}")

    # ── 5. Aggregate results ─────────────────────────────────────────────────
    avg_cem_dist    = float(np.mean([r["planned_tip_dist_to_goal"] for r in ep_results]))
    avg_random_dist = float(np.mean([r["random_tip_dist_to_goal"]  for r in ep_results]))
    avg_improvement = float(np.mean([r["improvement_over_random"]   for r in ep_results]))
    pct_better      = float(np.mean([r["improvement_over_random"] > 0 for r in ep_results])) * 100

    print(f"\n=== CEM PLANNING RESULTS ({n_episodes} episodes) ===")
    print(f"  Avg tip dist (CEM):    {avg_cem_dist:.4f}")
    print(f"  Avg tip dist (random): {avg_random_dist:.4f}")
    print(f"  Avg improvement:       {avg_improvement:+.4f}")
    print(f"  CEM better than random: {pct_better:.0f}% of episodes")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    # Plot 1: CEM cost convergence curve (averaged)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    mean_curve = np.mean(all_cost_curves, axis=0)
    std_curve  = np.std(all_cost_curves,  axis=0)
    xs = np.arange(1, n_cem_iters + 1)
    ax.plot(xs, mean_curve, color="#4fc3f7", linewidth=2, label="Mean cost")
    ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.25, color="#4fc3f7")
    ax.set_xlabel("CEM iteration", color="white"); ax.set_ylabel("Latent cost ||ẑ_T - z_goal||²", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.set_title("CEM Convergence in V-JEPA 2 Latent Space\n"
                 f"reacher-easy, horizon={horizon}, N={n_candidates}, {n_episodes} episodes",
                 color="white")
    ax.legend(facecolor="#222", labelcolor="white")
    plt.tight_layout()
    plt.savefig("/output/cem_cost_convergence.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # Plot 2: CEM vs Random bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    bars = ax.bar(
        ["CEM Planner\n(latent space)", "Random Policy\n(baseline)"],
        [avg_cem_dist, avg_random_dist],
        color=["#66bb6a", "#ef5350"], width=0.5,
        edgecolor="white", linewidth=0.7,
    )
    for bar, val in zip(bars, [avg_cem_dist, avg_random_dist]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", color="white",
                fontsize=13, fontweight="bold")
    ax.set_ylabel("Fingertip distance to goal (m)", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.set_title(
        f"CEM vs Random — Reacher-Easy Goal Reaching\n"
        f"V-JEPA 2 latent planning, {n_episodes} episodes · {pct_better:.0f}% episodes CEM < random",
        color="white"
    )
    plt.tight_layout()
    plt.savefig("/output/cem_vs_random.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── 7. Save JSON results ──────────────────────────────────────────────────
    final = {
        "n_episodes":             n_episodes,
        "horizon":                horizon,
        "n_candidates":           n_candidates,
        "n_elites":               n_elites,
        "n_cem_iters":            n_cem_iters,
        "avg_cem_tip_dist":       avg_cem_dist,
        "avg_random_tip_dist":    avg_random_dist,
        "avg_improvement_m":      avg_improvement,
        "pct_episodes_cem_better": pct_better,
        "episodes":               ep_results,
    }
    with open("/output/cem_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()

    print("\nSaved: cem_results.json, cem_cost_convergence.png, cem_vs_random.png")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Phase 3: CEM Goal-Reaching Planner")
    print("=" * 60)

    results = run_cem_planner.remote(
        n_episodes=10,
        horizon=10,
        n_candidates=512,
        n_elites=64,
        n_cem_iters=10,
        action_dim=2,         # reacher-easy
    )

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        if k == "episodes":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Download outputs
    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    for fname in ["cem_results.json", "cem_cost_convergence.png", "cem_vs_random.png",
                  "cem_episode_0.mp4", "cem_episode_1.mp4"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)],
                check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception:
            print(f"  Run manually: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")

    print("\nDone!")
