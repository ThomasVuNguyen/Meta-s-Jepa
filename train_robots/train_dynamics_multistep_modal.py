"""
Phase 4c — Multi-Step Rollout Dynamics Training + CEM Evaluation
================================================================
Runs on Modal A10G (~15-20 min, ~$0.50).

Two stages:
  1. Re-train the dynamics predictor with multi-step rollout loss.
     Instead of only penalizing 1-step prediction error, we unroll
     the model for H steps and backprop through the full chain:
       z1_pred = f(z0, a0)           → loss vs z1  (ground truth)
       z2_pred = f(z1_pred, a1)      → loss vs z2  (uses PREDICTED z!)
       z3_pred = f(z2_pred, a2)      → loss vs z3
     This directly penalizes compound error.

  2. Run CEM-based MPC evaluation on dm_control reacher-easy.
     Compares latent distance improvement against Phase 4b baseline.

Output: Updated dynamics model + CEM evaluation metrics.
"""

import modal
from pathlib import Path
import json

app = modal.App("vjepa2-phase4c")

model_cache = modal.Volume.from_name("vjepa2-model-cache", create_if_missing=True)
demo_vol    = modal.Volume.from_name("vjepa2-robot-demos", create_if_missing=True)
results_vol = modal.Volume.from_name("vjepa2-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libosmesa6-dev", "libosmesa6",
        "libgl1-mesa-glx", "libglib2.0-0",
        "patchelf", "gcc",
    ])
    .pip_install([
        "dm_control", "mujoco>=3.0.0",
        "transformers", "huggingface_hub", "safetensors",
        "torch", "torchvision", "numpy", "Pillow", "tqdm",
        "matplotlib", "imageio[ffmpeg]",
    ])
    .env({
        "MUJOCO_GL": "osmesa",
        "TRANSFORMERS_CACHE": "/cache/hf",
    })
)


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMICS PREDICTOR (same architecture as train_dynamics.py)
# ═══════════════════════════════════════════════════════════════════════════

def make_dynamics_model():
    """Create the DynamicsPredictor model (must be defined here for Modal)."""
    import torch
    import torch.nn as nn

    class DynamicsPredictor(nn.Module):
        def __init__(self, latent_dim=1024, action_dim=2, hidden_dim=512):
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
            delta_z = self.net(x)
            return z_t + delta_z

    return DynamicsPredictor


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: MULTI-STEP ROLLOUT TRAINING
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_multistep():
    """Train dynamics model with multi-step rollout loss."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    BATCH_SIZE = 64        # episodes per batch
    EPOCHS = 150
    LR = 3e-4
    ROLLOUT_H = 5          # unroll horizon
    EP_LEN = 200           # steps per episode
    N_EPISODES = 500

    # Load data
    print("Loading dataset...")
    data = np.load("/demos/reacher_easy_dynamics_demos.npz")
    z_all = np.concatenate([data['z_t'], data['z_next'][-1:]], axis=0)  # won't work — flat
    # Actually, the data is [N, 1024] where N = 500*200 = 100,000
    # Each episode is indices [ep*200 : (ep+1)*200]
    # z_t[i] and z_next[i] form a pair, and z_next[i] == z_t[i+1] within the same episode.
    z_t_all = torch.tensor(data['z_t'], dtype=torch.float32)     # [100000, 1024]
    a_t_all = torch.tensor(data['a_t'], dtype=torch.float32)     # [100000, 2]
    z_next_all = torch.tensor(data['z_next'], dtype=torch.float32) # [100000, 1024]

    # Reshape into episodes: [500, 200, dim]
    z_t_eps = z_t_all.view(N_EPISODES, EP_LEN, -1)       # [500, 200, 1024]
    a_t_eps = a_t_all.view(N_EPISODES, EP_LEN, -1)       # [500, 200, 2]
    z_next_eps = z_next_all.view(N_EPISODES, EP_LEN, -1)  # [500, 200, 1024]

    # Build target trajectory: [ep, step, 1024]
    # For rollout from step t, we need z_t, a_t, a_{t+1}, ..., a_{t+H-1}
    # and ground truth z_{t+1}, z_{t+2}, ..., z_{t+H}

    # Train/val split: 400 train, 100 val
    train_z = z_t_eps[:400].to(DEVICE)
    train_a = a_t_eps[:400].to(DEVICE)
    train_z_next = z_next_eps[:400].to(DEVICE)

    val_z = z_t_eps[400:].to(DEVICE)
    val_a = a_t_eps[400:].to(DEVICE)
    val_z_next = z_next_eps[400:].to(DEVICE)

    DynamicsPredictor = make_dynamics_model()
    model = DynamicsPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nMulti-Step Rollout Training (H={ROLLOUT_H})")
    print(f"Train episodes: {len(train_z)} | Val episodes: {len(val_z)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        # Random episode order
        perm = torch.randperm(len(train_z))
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_z), BATCH_SIZE):
            batch_idx = perm[batch_start:batch_start + BATCH_SIZE]
            bz = train_z[batch_idx]        # [B, 200, 1024]
            ba = train_a[batch_idx]        # [B, 200, 2]
            bz_next = train_z_next[batch_idx]  # [B, 200, 1024]

            # Random start timestep (ensure room for H-step rollout)
            max_start = EP_LEN - ROLLOUT_H
            t0 = torch.randint(0, max_start, (1,)).item()

            # Unroll dynamics for H steps
            z_pred = bz[:, t0, :]  # [B, 1024]
            loss = torch.tensor(0.0, device=DEVICE)

            for h in range(ROLLOUT_H):
                a_h = ba[:, t0 + h, :]          # [B, 2]
                z_pred = model(z_pred, a_h)      # [B, 1024] — uses predicted z!
                z_true = bz_next[:, t0 + h, :]   # [B, 1024]

                # Linearly increasing weight: later steps penalized more
                weight = (h + 1) / ROLLOUT_H
                step_loss = nn.functional.smooth_l1_loss(z_pred, z_true)
                loss = loss + weight * step_loss

            loss = loss / ROLLOUT_H  # normalize

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        train_losses.append(epoch_loss)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_start in range(0, len(val_z), BATCH_SIZE):
                bz = val_z[batch_start:batch_start + BATCH_SIZE]
                ba = val_a[batch_start:batch_start + BATCH_SIZE]
                bz_next = val_z_next[batch_start:batch_start + BATCH_SIZE]

                # Evaluate at multiple start points
                for t0 in range(0, EP_LEN - ROLLOUT_H, ROLLOUT_H * 2):
                    z_pred = bz[:, t0, :]
                    batch_loss = 0.0
                    for h in range(ROLLOUT_H):
                        z_pred = model(z_pred, ba[:, t0 + h, :])
                        z_true = bz_next[:, t0 + h, :]
                        weight = (h + 1) / ROLLOUT_H
                        batch_loss += weight * nn.functional.smooth_l1_loss(z_pred, z_true).item()
                    val_loss += batch_loss / ROLLOUT_H
                    n_val += 1

        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/results/dynamics_multistep.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {epoch_loss:.5f} | Val: {val_loss:.5f} | "
                  f"Best: {best_val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.5f}")

    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (multi-step)")
    plt.plot(val_losses, label="Val Loss (multi-step)")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.title(f"Dynamics Predictor — {ROLLOUT_H}-Step Rollout Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("/results/dynamics_multistep_loss.png", dpi=150)
    print("Saved loss curve to /results/dynamics_multistep_loss.png")

    results_vol.commit()
    return {"best_val_loss": best_val_loss, "epochs": EPOCHS, "rollout_h": ROLLOUT_H}


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: CEM PLANNER EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def run_cem_eval():
    """Run CEM-based MPC evaluation with the multi-step-trained dynamics model."""
    import torch
    import numpy as np
    from dm_control import suite
    from transformers import AutoModel
    from PIL import Image
    import torchvision.transforms as T
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"

    vjepa_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    @torch.no_grad()
    def encode_image(model, img):
        x = vjepa_transform(img)
        window = [x] * 8
        clips_t = torch.stack(window).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        outputs = model(pixel_values_videos=clips_t, return_dict=True)
        return outputs.last_hidden_state.mean(dim=1)  # [1, 1024]

    # --- CEM Planner ---
    class CEMPlanner:
        def __init__(self, dynamics_model, num_samples=500, horizon=10,
                     action_dim=2, num_iterations=5, elite_frac=0.1):
            self.dynamics_model = dynamics_model
            self.num_samples = num_samples
            self.horizon = horizon
            self.action_dim = action_dim
            self.num_iterations = num_iterations
            self.num_elites = max(1, int(num_samples * elite_frac))
            self._prev_mean = None

        @torch.no_grad()
        def plan(self, z_t, z_goal):
            if self._prev_mean is not None:
                mean = torch.zeros(self.horizon, self.action_dim, device=DEVICE)
                mean[:-1] = self._prev_mean[1:]
            else:
                mean = torch.zeros(self.horizon, self.action_dim, device=DEVICE)
            std = torch.ones(self.horizon, self.action_dim, device=DEVICE) * 0.5

            best_actions = None
            best_cost = float('inf')

            for _ in range(self.num_iterations):
                noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=DEVICE)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1, 1)

                costs = self._evaluate(actions, z_t, z_goal)
                elite_idx = torch.topk(costs, self.num_elites, largest=False).indices
                elite_actions = actions[elite_idx]

                if costs[elite_idx[0]].item() < best_cost:
                    best_cost = costs[elite_idx[0]].item()
                    best_actions = actions[elite_idx[0]]

                mean = elite_actions.mean(dim=0)
                std = elite_actions.std(dim=0).clamp(min=0.05)

            self._prev_mean = mean.clone()
            return best_actions[0].cpu().numpy()

        def _evaluate(self, actions, z_t, z_goal):
            z_curr = z_t.expand(self.num_samples, -1)
            z_goal_exp = z_goal.expand(self.num_samples, -1)
            total_cost = torch.zeros(self.num_samples, device=DEVICE)
            for t in range(self.horizon):
                z_curr = self.dynamics_model(z_curr, actions[:, t, :])
                weight = (t + 1) / self.horizon
                total_cost += weight * torch.norm(z_curr - z_goal_exp, dim=-1)
            return total_cost

    # --- Load models ---
    print("Loading V-JEPA 2...")
    vjepa_model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True,
        cache_dir="/cache/hf"
    ).to(DEVICE)
    vjepa_model.eval()

    print("Loading multi-step dynamics model...")
    DynamicsPredictor = make_dynamics_model()
    dynamics_model = DynamicsPredictor().to(DEVICE)
    dynamics_model.load_state_dict(torch.load("/results/dynamics_multistep.pt", weights_only=True))
    dynamics_model.eval()

    # --- Generate expert goal ---
    print("Generating goal with P-controller...")
    raw_env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': 42})
    ts = raw_env.reset()
    obs = ts.observation
    best_dist = float('inf')
    best_frame = None

    for step in range(300):
        action = np.clip(obs["to_target"] * 5.0, -1, 1).astype(np.float32)
        ts = raw_env.step(action)
        obs = ts.observation
        dist = np.linalg.norm(obs["to_target"])
        if dist < best_dist:
            best_dist = dist
            best_frame = raw_env.physics.render(height=224, width=224, camera_id=0).copy()

    goal_img = Image.fromarray(best_frame)
    z_goal = encode_image(vjepa_model, goal_img)
    print(f"  Best to_target dist: {best_dist:.4f} | Goal latent norm: {z_goal.norm().item():.2f}")

    # --- Init eval env ---
    print("Initializing eval env...")
    from dm_control.suite.wrappers import pixels
    env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': 42})
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs={
        'height': 224, 'width': 224, 'camera_id': 0
    })

    planner = CEMPlanner(dynamics_model, num_samples=500, horizon=10,
                         num_iterations=5, elite_frac=0.1)

    # --- Run evaluation ---
    print("\n" + "=" * 60)
    print("CEM-MPC Evaluation (Phase 4c: multi-step dynamics)")
    print("=" * 60)

    time_step = env.reset()
    init_img = Image.fromarray(time_step.observation['pixels'])
    z_init = encode_image(vjepa_model, init_img)
    init_latent_dist = torch.norm(z_init - z_goal).item()
    print(f"Initial latent distance: {init_latent_dist:.2f}")

    frames = []
    latent_distances = []
    rewards_log = []
    total_reward = 0.0
    max_steps = 200

    for step in range(max_steps):
        img_t = Image.fromarray(time_step.observation['pixels'])
        frames.append(np.array(img_t))
        z_t = encode_image(vjepa_model, img_t)

        lat_dist = torch.norm(z_t - z_goal).item()
        latent_distances.append(lat_dist)

        action = planner.plan(z_t, z_goal)
        time_step = env.step(action)
        reward = time_step.reward or 0.0
        total_reward += reward
        rewards_log.append(reward)

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}/{max_steps} | "
                  f"Action: [{action[0]:5.2f}, {action[1]:5.2f}] | "
                  f"Latent Dist: {lat_dist:.2f} | "
                  f"Env Reward: {reward:.3f}")

    # Final
    final_img = Image.fromarray(time_step.observation['pixels'])
    z_final = encode_image(vjepa_model, final_img)
    final_latent_dist = torch.norm(z_final - z_goal).item()

    results = {
        "initial_latent_dist": init_latent_dist,
        "final_latent_dist": final_latent_dist,
        "min_latent_dist": min(latent_distances),
        "improvement_pct": (1 - final_latent_dist / init_latent_dist) * 100,
        "total_env_reward": total_reward,
        "max_step_reward": max(rewards_log),
        "phase": "4c",
    }

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete.")
    print(f"  Initial Latent Dist:  {results['initial_latent_dist']:.2f}")
    print(f"  Final Latent Dist:    {results['final_latent_dist']:.2f}")
    print(f"  Min Latent Dist:      {results['min_latent_dist']:.2f}")
    print(f"  Improvement:          {results['improvement_pct']:.1f}%")
    print(f"  Total Env Reward:     {results['total_env_reward']:.2f}")
    print(f"{'=' * 60}")

    # Save outputs
    imageio.mimwrite("/results/mpc_cem_4c_rollout.mp4", frames, fps=30)
    print("Saved rollout video.")

    with open("/results/phase4c_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(latent_distances, color='#4361ee', linewidth=1.5)
    axes[0].axhline(y=init_latent_dist, color='red', linestyle='--', alpha=0.5, label='Initial')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Latent Distance to Goal")
    axes[0].set_title("Phase 4c: Latent Distance (multi-step dynamics)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.cumsum(rewards_log), color='#f72585', linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative Env Reward")
    axes[1].set_title("Phase 4c: Cumulative Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/results/mpc_cem_4c_metrics.png", dpi=150)
    print("Saved metrics plot.")

    # Save latent distances as numpy for comparison
    np.save("/results/latent_distances_4c.npy", np.array(latent_distances))

    results_vol.commit()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    import subprocess

    print("=" * 70)
    print("Phase 4c: Multi-Step Rollout Training + CEM Evaluation")
    print("=" * 70)

    # Stage 1: Train
    print("\n[Stage 1] Training dynamics model with multi-step rollout loss...")
    train_result = train_multistep.remote()
    print(f"\nTraining result: {train_result}")

    # Stage 2: Evaluate
    print("\n[Stage 2] Running CEM-MPC evaluation...")
    eval_result = run_cem_eval.remote()
    print(f"\nEvaluation result: {eval_result}")

    # Download results locally
    out_dir = Path("train_robots/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "dynamics_multistep.pt",
        "dynamics_multistep_loss.png",
        "mpc_cem_4c_rollout.mp4",
        "mpc_cem_4c_metrics.png",
        "phase4c_results.json",
        "latent_distances_4c.npy",
    ]:
        p = str(out_dir / fname)
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-results", fname, p],
                check=True
            )
            print(f"  Downloaded → {p}")
        except Exception:
            print(f"  ⚠ Run manually: modal volume get vjepa2-results {fname} {p}")

    # Print comparison
    print("\n" + "=" * 70)
    print("Phase 4b vs 4c Comparison")
    print("=" * 70)
    print(f"  Phase 4b: 41.3% latent dist improvement, 6.0 env reward")
    print(f"  Phase 4c: {eval_result['improvement_pct']:.1f}% latent dist improvement, "
          f"{eval_result['total_env_reward']:.1f} env reward")
