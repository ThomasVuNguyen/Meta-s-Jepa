"""
V-JEPA 2 Receding-Horizon MPC — Phase 4 of Meta-s-Jepa
=========================================================
Three improvements over Phase 3 (open-loop CEM, 60% win rate, T=10):

  1. Goal-conditioned training data
       Collect rollouts using a proportional controller that steers the
       reacher arm toward the target → gives the dynamics MLP experience
       with directed transitions, not just random ones.

  2. Fine-tune dynamics MLP on combined data
       Mix goal-directed + cached random rollouts.
       Same 3-layer MLP architecture, fine-tuned from Phase 2 checkpoint.

  3. Receding-horizon MPC (replan every step)
       Instead of executing T actions open-loop, plan T steps but execute
       only the first action, observe the new frame, embed it, replan.
       Eliminates accumulation of model prediction error.

Hypothesis: MPC replanning + goal-conditioned data → win rate > 80%
            and average tip distance < 0.10 m (vs 0.220 m in Phase 3)

Compute estimate:
  Stage 1: Goal rollout collection (CPU, ~15 min, ~$0.05)
  Stage 2: V-JEPA embed + fine-tune (A10G, ~35 min, ~$0.64)
  Stage 3: MPC evaluation 10 eps × 50 steps × replan (A10G, ~30 min, ~$0.55)
  Total: ~80 min, ~$1.24
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-mpc")

model_cache  = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol   = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol  = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Collect goal-directed rollouts for reacher-easy
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image, cpu=4, memory=8192, timeout=1800,
    volumes={"/rollouts": rollout_vol},
)
def collect_goal_directed_rollouts(
    n_episodes: int = 50,
    episode_len: int = 200,
    render_size: int = 256,
    p_goal: float = 0.75,   # probability of using goal-seeking action
    force_recollect: bool = False,
):
    """
    Collect reacher-easy rollouts with a proportional controller.

    Policy: with probability p_goal, apply an action that moves the
    fingertip toward the target (estimated via Jacobian transpose).
    With probability 1-p_goal, apply a random action for exploration.
    This produces directed transitions that a random policy never sees.
    """
    import os, json
    import numpy as np
    from pathlib import Path

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    from dm_control import suite

    save_dir     = Path("/rollouts/reacher_easy_goal")
    frames_path  = save_dir / "frames.npy"
    actions_path = save_dir / "actions.npy"
    meta_path    = save_dir / "meta.json"

    if frames_path.exists() and not force_recollect:
        meta = json.loads(meta_path.read_text())
        print(f"[CACHE] reacher_easy_goal: {meta['n_transitions']} transitions already collected")
        return meta

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[COLLECT] reacher_easy_goal — {n_episodes} episodes × {episode_len} steps")
    print(f"  Goal-seeking probability: {p_goal:.0%}")

    env         = suite.load("reacher", "easy", task_kwargs={"random": 42})
    action_spec = env.action_spec()
    rng         = np.random.RandomState(0)

    all_frames  = []
    all_actions = []

    for ep in range(n_episodes):
        ts = env.reset()

        ep_frames  = []
        ep_actions = []

        for step in range(episode_len):
            # Render current frame
            frame = env.physics.render(height=render_size, width=render_size, camera_id=0)

            # Goal-directed action via Jacobian transpose
            if rng.rand() < p_goal:
                try:
                    finger_pos = env.physics.named.data.geom_xpos["finger", :2].copy()
                    target_pos = env.physics.named.data.geom_xpos["target", :2].copy()
                    error      = target_pos - finger_pos  # 2D direction error

                    # Jacobian of fingertip w.r.t. joint positions (2×2 for 2-DOF)
                    jac_full = env.physics.data.jacp.reshape(3, -1)[:2, :2]
                    torques  = jac_full.T @ error
                    norm     = np.linalg.norm(torques)
                    if norm > 1e-6:
                        action = np.clip(torques / norm, -1.0, 1.0).astype(np.float32)
                    else:
                        action = rng.uniform(-1, 1, size=2).astype(np.float32)
                except Exception:
                    action = rng.uniform(-1, 1, size=2).astype(np.float32)
            else:
                action = rng.uniform(
                    action_spec.minimum, action_spec.maximum
                ).astype(np.float32)

            ts = env.step(action)
            ep_frames.append(frame)
            ep_actions.append(action)

            if ts.last():
                break

        # Append consecutive pairs (skip last frame)
        for t in range(len(ep_frames) - 1):
            all_frames.append(ep_frames[t])
            all_actions.append(ep_actions[t])

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} done  ({len(all_frames)} pairs so far)")

    frames_arr  = np.array(all_frames,  dtype=np.uint8)
    actions_arr = np.array(all_actions, dtype=np.float32)
    n_transitions = len(frames_arr)
    print(f"  → {n_transitions} goal-directed transitions collected")

    np.save(str(frames_path),  frames_arr)
    np.save(str(actions_path), actions_arr)
    meta = {
        "env_key": "reacher_easy_goal",
        "domain": "reacher", "task": "easy",
        "n_transitions": n_transitions,
        "action_dim": 2, "render_size": render_size,
        "p_goal": p_goal,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    rollout_vol.commit()
    print(f"  ✓ Saved to /rollouts/reacher_easy_goal/")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 + 3: Fine-tune dynamics MLP + Run MPC
# ─────────────────────────────────────────────────────────────────────────────

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
def finetune_and_mpc(
    # Fine-tuning
    hidden_dim: int = 512,
    n_layers: int = 3,
    ft_lr: float = 3e-4,
    ft_epochs: int = 30,
    batch_size: int = 256,
    # MPC
    n_episodes: int = 10,
    mpc_steps: int = 50,      # total env steps per episode
    replan_every: int = 1,    # replan every N steps (1 = replan each step)
    horizon: int = 25,        # planning horizon T
    n_candidates: int = 256,  # CEM candidates (smaller since we replan often)
    n_elites: int = 32,
    n_cem_iters: int = 5,
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
    import imageio

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

    # ── Load / Define dynamics MLP ─────────────────────────────────────────
    print("[2] Loading dynamics MLP from Phase 2...")
    ckpt = torch.load("/output/dynamics_mlp.pt", map_location=DEVICE)
    z_dim       = ckpt["z_dim"]          # 1024
    a_pad_dim   = ckpt["max_action_dim"] # 6
    input_dim   = z_dim + a_pad_dim      # 1030

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
    print(f"    Loaded Phase 2 checkpoint: val_loss={ckpt['final_val_loss']:.4f}")

    def pad_action(a_np, pad_to=6):
        pad = np.zeros(pad_to - len(a_np), dtype=np.float32)
        return np.concatenate([a_np, pad])

    # ── Load goal-directed rollout data ───────────────────────────────────
    print("\n[3] Loading goal-directed rollouts and extracting embeddings...")
    gd_dir       = Path("/rollouts/reacher_easy_goal")
    frames_gd    = np.load(str(gd_dir / "frames.npy"))    # (N, H, W, 3)
    actions_gd   = np.load(str(gd_dir / "actions.npy"))   # (N, 2)
    meta_gd      = json.loads((gd_dir / "meta.json").read_text())
    N_gd         = len(frames_gd)
    print(f"    {N_gd} goal-directed transitions from reacher_easy_goal")

    print("    Extracting z_t embeddings...")
    z_gd = embed_batch(frames_gd)           # (N, 1024)
    print("    Extracting z_{t+1} embeddings...")
    z_gd1 = embed_batch(frames_gd[1:])      # (N-1, 1024)

    Z_t  = z_gd[:-1].numpy()               # (N-1, 1024)
    Z_t1 = z_gd1.numpy()                   # (N-1, 1024)

    # Pad actions to 6-d
    A_t  = np.array([pad_action(a) for a in actions_gd[:-1]], dtype=np.float32)  # (N-1, 6)

    # Also load original random reacher_easy rollouts for mixing
    rand_dir = Path("/rollouts/reacher_easy")
    if rand_dir.exists():
        frames_rand  = np.load(str(rand_dir / "frames.npy"))
        actions_rand = np.load(str(rand_dir / "actions.npy"))
        print(f"    Mixing in {len(frames_rand)} random transitions from Phase 2")
        zr   = embed_batch(frames_rand).numpy()[:-1]
        zr1  = embed_batch(frames_rand[1:]).numpy()
        Ar   = np.array([pad_action(a) for a in actions_rand[:-1]], dtype=np.float32)
        # Combine goal-directed (primary) + random (auxiliary)
        Z_t  = np.concatenate([Z_t, zr],  axis=0)
        Z_t1 = np.concatenate([Z_t1, zr1], axis=0)
        A_t  = np.concatenate([A_t, Ar],  axis=0)

    N = len(Z_t)
    print(f"    Combined dataset: {N} transitions (goal-directed + random)")

    # ── Fine-tune dynamics MLP ────────────────────────────────────────────
    print(f"\n[4] Fine-tuning dynamics MLP ({ft_epochs} epochs, lr={ft_lr})...")
    rng_np    = np.random.RandomState(42)
    perm      = rng_np.permutation(N)
    split     = int(0.85 * N)
    tr_idx, te_idx = perm[:split], perm[split:]

    Z_t_tr  = torch.tensor(Z_t[tr_idx]).to(DEVICE)
    Z_t1_tr = torch.tensor(Z_t1[tr_idx]).to(DEVICE)
    A_t_tr  = torch.tensor(A_t[tr_idx]).to(DEVICE)
    Z_t_te  = torch.tensor(Z_t[te_idx]).to(DEVICE)
    Z_t1_te = torch.tensor(Z_t1[te_idx]).to(DEVICE)
    A_t_te  = torch.tensor(A_t[te_idx]).to(DEVICE)

    dataset   = TensorDataset(Z_t_tr, A_t_tr, Z_t1_tr)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=ft_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    criterion = nn.MSELoss()

    ft_train_losses = []
    ft_val_losses   = []

    for epoch in range(ft_epochs):
        dynamics.train()
        ep_loss = 0.0
        for z_b, a_b, z1_b in loader:
            optimizer.zero_grad()
            loss = criterion(dynamics(z_b, a_b), z1_b)
            loss.backward()
            nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(z_b)
        scheduler.step()
        ft_train_losses.append(ep_loss / len(tr_idx))

        dynamics.eval()
        with torch.no_grad():
            ft_val_losses.append(criterion(dynamics(Z_t_te, A_t_te), Z_t1_te).item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{ft_epochs}  train={ft_train_losses[-1]:.4f}  val={ft_val_losses[-1]:.4f}")

    print(f"  Final val_loss after fine-tuning: {ft_val_losses[-1]:.4f}  "
          f"(was {ckpt['final_val_loss']:.4f})")

    # Save fine-tuned model
    torch.save({
        **ckpt,
        "model_state":    dynamics.state_dict(),
        "final_val_loss": ft_val_losses[-1],
        "phase":          4,
        "ft_epochs":      ft_epochs,
        "ft_lr":          ft_lr,
    }, "/output/dynamics_mlp_ft.pt")
    output_vol.commit()

    # Fine-tune loss chart
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    ax.plot(ft_train_losses, color="#4fc3f7", label="Train", linewidth=1.5)
    ax.plot(ft_val_losses,   color="#ff8a65", label="Val",   linewidth=1.5)
    ax.axhline(ckpt["final_val_loss"], color="#aaa", linestyle="--", linewidth=1,
               label=f"Phase 2 baseline ({ckpt['final_val_loss']:.4f})")
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("MSE Loss", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white")
    ax.set_title("Phase 4: Fine-Tuning on Goal-Directed Data", color="white")
    plt.tight_layout()
    plt.savefig("/output/mpc_finetune_loss.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── MPC Evaluation ────────────────────────────────────────────────────
    print(f"\n[5] MPC evaluation — {n_episodes} episodes, {mpc_steps} steps, replan every {replan_every}")
    from dm_control import suite as dmc_suite

    dynamics.eval()

    def cem_plan_single(z_start_t, z_goal_t, rng_local):
        """CEM planning step — returns the first action to execute."""
        T = horizon; N_c = n_candidates; K = n_elites; I = n_cem_iters
        mu  = np.zeros((T, a_pad_dim), dtype=np.float32)
        sig = np.ones((T, a_pad_dim), dtype=np.float32)
        z_s = z_start_t.unsqueeze(0).expand(N_c, -1)
        z_g = z_goal_t.unsqueeze(0).expand(N_c,  -1)
        best_action = None
        best_cost   = float("inf")

        for _ in range(I):
            eps      = rng_local.randn(N_c, T, a_pad_dim).astype(np.float32)
            act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
            act_t    = torch.tensor(act_seqs, device=DEVICE)
            z_curr   = z_s.clone()
            with torch.no_grad():
                for t in range(T):
                    z_curr = dynamics(z_curr, act_t[:, t, :])
            costs      = ((z_curr - z_g) ** 2).sum(dim=-1).cpu().numpy()
            elite_idx  = np.argsort(costs)[:K]
            elite_seqs = act_seqs[elite_idx]
            mu  = elite_seqs.mean(axis=0)
            sig = elite_seqs.std(axis=0) + 1e-6
            if costs[elite_idx[0]] < best_cost:
                best_cost   = costs[elite_idx[0]]
                best_action = act_seqs[elite_idx[0], 0, :]  # first action
        return best_action, best_cost

    env      = dmc_suite.load("reacher", "easy", task_kwargs={"random": 0})
    env_goal = dmc_suite.load("reacher", "easy", task_kwargs={"random": 1})
    aspec    = env.action_spec()
    rng_eval = np.random.RandomState(99)

    ep_results    = []
    all_tip_dists = []  # (n_episodes, mpc_steps) for trajectory plots
    all_frames_ep = []  # store frames for video

    for ep_idx in range(n_episodes):
        print(f"\n  Episode {ep_idx + 1}/{n_episodes}")
        ts = env.reset()

        # Sample random goal state
        env_goal.reset()
        for _ in range(rng_eval.randint(5, 25)):
            env_goal.step(rng_eval.uniform(-1, 1, size=2))
        frame_goal  = env_goal.physics.render(height=256, width=256, camera_id=0)
        z_goal_dev  = embed_single(frame_goal).to(DEVICE)
        target_pos  = env_goal.physics.named.data.geom_xpos["target", :2].copy()

        ep_frames    = []
        ep_tip_dists = []
        plan_cache   = None  # cache plan between replanning steps

        for step in range(mpc_steps):
            frame_curr = env.physics.render(height=256, width=256, camera_id=0)
            ep_frames.append(frame_curr)
            tip_pos = env.physics.named.data.geom_xpos["finger", :2].copy()
            ep_tip_dists.append(float(np.linalg.norm(tip_pos - target_pos)))

            # Replan every `replan_every` steps
            if step % replan_every == 0:
                z_curr_dev = embed_single(frame_curr).to(DEVICE)
                plan_cache, _ = cem_plan_single(z_curr_dev, z_goal_dev, rng_eval)

            # Extract action for this step (first action of plan)
            a_full   = plan_cache[:2]  # back to action_dim=2
            a_exec   = np.clip(a_full, aspec.minimum, aspec.maximum)
            ts       = env.step(a_exec)

        final_tip = env.physics.named.data.geom_xpos["finger", :2].copy()
        final_dist = float(np.linalg.norm(final_tip - target_pos))
        ep_tip_dists.append(final_dist)
        all_tip_dists.append(ep_tip_dists)
        all_frames_ep.append(ep_frames)

        # Random baseline for same episode
        ts = env.reset()
        for _ in range(mpc_steps):
            env.step(rng_eval.uniform(aspec.minimum, aspec.maximum))
        rand_final = env.physics.named.data.geom_xpos["finger", :2].copy()
        random_dist = float(np.linalg.norm(rand_final - target_pos))

        ep_result = {
            "episode":    ep_idx,
            "mpc_dist":   final_dist,
            "random_dist": random_dist,
            "min_dist":   float(min(ep_tip_dists)),
            "improvement": random_dist - final_dist,
        }
        ep_results.append(ep_result)
        print(f"    MPC final dist: {final_dist:.4f}  Random: {random_dist:.4f}  "
              f"Best during ep: {ep_result['min_dist']:.4f}  Δ={ep_result['improvement']:+.4f}")

        # Save video for first 3 episodes
        if ep_idx < 3:
            vpath = f"/output/mpc_episode_{ep_idx}.mp4"
            writer = imageio.get_writer(vpath, fps=15)
            for fr in ep_frames:
                writer.append_data(fr)
            writer.close()

    # ── Aggregate ──────────────────────────────────────────────────────────
    avg_mpc_dist    = float(np.mean([r["mpc_dist"]    for r in ep_results]))
    avg_random_dist = float(np.mean([r["random_dist"] for r in ep_results]))
    avg_min_dist    = float(np.mean([r["min_dist"]    for r in ep_results]))
    pct_better      = float(np.mean([r["improvement"] > 0 for r in ep_results])) * 100

    print(f"\n=== MPC RESULTS ({n_episodes} episodes, {mpc_steps} steps, replan every {replan_every}) ===")
    print(f"  Avg final dist (MPC):        {avg_mpc_dist:.4f} m")
    print(f"  Avg final dist (random):     {avg_random_dist:.4f} m")
    print(f"  Avg min dist during episode: {avg_min_dist:.4f} m")
    print(f"  MPC better than random:      {pct_better:.0f}% of episodes")
    print(f"  Phase 3 CEM avg dist:        0.2200 m (baseline)")
    print(f"  Improvement vs Phase 3:      {0.2200 - avg_mpc_dist:+.4f} m")

    # ── Plots ─────────────────────────────────────────────────────────────
    # 1. Tip distance trajectory over time (all episodes)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    for i, traj in enumerate(all_tip_dists):
        alpha = 0.3 if i < n_episodes - 1 else 1.0
        ax.plot(traj, alpha=alpha, linewidth=(0.8 if i < n_episodes - 1 else 2.0),
                color="#4fc3f7")
    mean_traj = np.mean(all_tip_dists, axis=0)
    ax.plot(mean_traj, color="#ffa726", linewidth=2.5, label=f"Mean (n={n_episodes})")
    ax.axhline(avg_random_dist, color="#ef5350", linestyle="--", linewidth=1.5,
               label=f"Random baseline ({avg_random_dist:.3f} m)")
    ax.axhline(0.2200, color="#aaa", linestyle=":", linewidth=1,
               label="Phase 3 CEM (0.220 m)")
    ax.set_xlabel("Environment step", color="white")
    ax.set_ylabel("Fingertip → goal distance (m)", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white")
    ax.set_title(
        f"Phase 4: Receding-Horizon MPC – Tip Distance Trajectory\n"
        f"reacher-easy, replan every {replan_every} step, T={horizon}, N={n_candidates}",
        color="white"
    )
    plt.tight_layout()
    plt.savefig("/output/mpc_trajectory.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # 2. Summary bar chart vs Phase 3
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    labels = ["Phase 3\nCEM open-loop\n(T=10)", "Phase 4\nMPC replan\n(T=25)", "Random\nbaseline"]
    values = [0.2200, avg_mpc_dist, avg_random_dist]
    colors = ["#ffa726", "#66bb6a", "#ef5350"]
    bars   = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                f"{val:.3f} m", ha="center", va="bottom", color="white",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg fingertip distance to goal (m)", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.set_title(
        f"Phase 4 vs Phase 3 vs Random — Reacher-Easy\n"
        f"{n_episodes} episodes · {pct_better:.0f}% episodes MPC < random",
        color="white"
    )
    plt.tight_layout()
    plt.savefig("/output/mpc_summary.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── Save ──────────────────────────────────────────────────────────────
    final_results = {
        "phase": 4,
        "n_episodes":        n_episodes,
        "mpc_steps":         mpc_steps,
        "replan_every":      replan_every,
        "horizon":           horizon,
        "n_candidates":      n_candidates,
        "ft_final_val_loss": ft_val_losses[-1],
        "phase2_val_loss":   ckpt["final_val_loss"],
        "avg_mpc_dist_m":      avg_mpc_dist,
        "avg_random_dist_m":   avg_random_dist,
        "avg_min_dist_m":      avg_min_dist,
        "phase3_avg_dist_m":   0.2200,
        "improvement_vs_p3_m": 0.2200 - avg_mpc_dist,
        "pct_better_than_random": pct_better,
        "episodes": ep_results,
    }
    with open("/output/mpc_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    output_vol.commit()
    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Phase 4: Goal-Conditioned MPC")
    print("=" * 60)

    # Stage 1: Collect goal-directed rollouts (CPU)
    print("\n[Stage 1] Collecting goal-directed rollouts...")
    meta = collect_goal_directed_rollouts.remote(
        n_episodes=50, episode_len=200, p_goal=0.75,
        force_recollect=False,
    )
    print(f"  {meta['n_transitions']} goal-directed transitions ready")

    # Stage 2+3: Fine-tune + MPC (A10G)
    print("\n[Stage 2+3] Fine-tuning + MPC evaluation on A10G...")
    results = finetune_and_mpc.remote(
        ft_epochs=30, ft_lr=3e-4,
        n_episodes=10, mpc_steps=50, replan_every=1,
        horizon=25, n_candidates=256, n_elites=32, n_cem_iters=5,
    )

    print("\n=== PHASE 4 FINAL RESULTS ===")
    for k, v in results.items():
        if k == "episodes": continue
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        else:                    print(f"  {k}: {v}")

    # Download outputs
    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    for fname in ["mpc_results.json", "mpc_summary.png", "mpc_trajectory.png",
                  "mpc_finetune_loss.png", "mpc_episode_0.mp4", "mpc_episode_1.mp4"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)], check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception:
            print(f"  Run manually: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")

    print("\nDone!")
