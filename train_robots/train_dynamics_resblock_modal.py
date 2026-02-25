"""
Phase 4d — Residual Block Dynamics Architecture + CEM Evaluation
================================================================
Runs on Modal A10G (~15-20 min, ~$0.30).

Tests the hypothesis that the MLP dynamics model's limited capacity
(512 hidden dim, ~1.2M params) is the bottleneck in compounding error.

Architecture change:
  - Hidden dim: 512 → 1024 (matches latent dim)
  - 4 residual blocks with skip connections
  - ~4.2M params (3.5x more than Phase 3)
  - Back to 1-step loss (multi-step regressed in Phase 4c)

Output: Updated dynamics model + CEM evaluation metrics.
"""

import modal
from pathlib import Path
import json

app = modal.App("vjepa2-phase4d")

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
# DYNAMICS PREDICTOR — RESIDUAL BLOCK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

def make_dynamics_model():
    """Create the ResBlock DynamicsPredictor."""
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        """Residual block with LayerNorm and ReLU."""
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x + self.net(x))

    class DynamicsPredictor(nn.Module):
        """
        Wider MLP with residual blocks.
        Input: concat(z_t [1024], a_t [2]) → 1026
        Projection → 1024 → 4 ResBlocks → delta_z [1024]
        ~4.2M params
        """
        def __init__(self, latent_dim=1024, action_dim=2, hidden_dim=1024):
            super().__init__()
            # Project input (with action) to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            # 4 residual blocks
            self.blocks = nn.Sequential(
                ResBlock(hidden_dim),
                ResBlock(hidden_dim),
                ResBlock(hidden_dim),
                ResBlock(hidden_dim),
            )
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, z_t, a_t):
            x = torch.cat([z_t, a_t], dim=-1)
            h = self.input_proj(x)
            h = self.blocks(h)
            delta_z = self.output_proj(h)
            return z_t + delta_z  # residual prediction

    return DynamicsPredictor


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: TRAIN WITH 1-STEP LOSS (standard, not multi-step)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_resblock():
    """Train ResBlock dynamics model with 1-step loss."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    BATCH_SIZE = 256
    EPOCHS = 150
    LR = 3e-4

    # Load data (flat transitions)
    print("Loading dataset...")
    data = np.load("/demos/reacher_easy_dynamics_demos.npz")
    z_t    = torch.tensor(data['z_t'], dtype=torch.float32)
    a_t    = torch.tensor(data['a_t'], dtype=torch.float32)
    z_next = torch.tensor(data['z_next'], dtype=torch.float32)

    n = len(z_t)
    train_size = int(0.8 * n)

    train_ds = TensorDataset(z_t[:train_size], a_t[:train_size], z_next[:train_size])
    val_ds   = TensorDataset(z_t[train_size:], a_t[train_size:], z_next[train_size:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    DynamicsPredictor = make_dynamics_model()
    model = DynamicsPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nResBlock Dynamics Training")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Device: {DEVICE}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for z, a, zn in train_loader:
            z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
            optimizer.zero_grad()
            zn_pred = model(z, a)
            loss = criterion(zn_pred, zn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * z.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, zn in val_loader:
                z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
                zn_pred = model(z, a)
                loss = criterion(zn_pred, zn)
                val_loss += loss.item() * z.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/results/dynamics_resblock.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nDone. Best Val Loss: {best_val_loss:.6f}")

    # Compare: old MLP best was ~0.01, how does ResBlock compare?
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (ResBlock)")
    plt.plot(val_losses, label="Val Loss (ResBlock)")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.title("Dynamics Predictor — ResBlock (4.2M params) vs MLP (1.2M params)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/results/dynamics_resblock_loss.png", dpi=150)

    results_vol.commit()
    return {"best_val_loss": best_val_loss, "n_params": n_params, "epochs": EPOCHS}


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
    """Run CEM-based MPC evaluation with ResBlock dynamics model."""
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
        return outputs.last_hidden_state.mean(dim=1)

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

    print("Loading ResBlock dynamics model...")
    DynamicsPredictor = make_dynamics_model()
    dynamics_model = DynamicsPredictor().to(DEVICE)
    dynamics_model.load_state_dict(torch.load("/results/dynamics_resblock.pt", weights_only=True))
    dynamics_model.eval()
    print(f"  Params: {sum(p.numel() for p in dynamics_model.parameters()):,}")

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
    print(f"  Goal dist: {best_dist:.4f} | Latent norm: {z_goal.norm():.2f}")

    # --- Eval env ---
    from dm_control.suite.wrappers import pixels
    env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': 42})
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs={
        'height': 224, 'width': 224, 'camera_id': 0
    })

    planner = CEMPlanner(dynamics_model, num_samples=500, horizon=10,
                         num_iterations=5, elite_frac=0.1)

    # --- Run ---
    print("\n" + "=" * 60)
    print("CEM-MPC Evaluation (Phase 4d: ResBlock dynamics)")
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
                  f"Latent Dist: {lat_dist:.2f} | Reward: {reward:.3f}")

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
        "phase": "4d",
    }

    print(f"\n{'=' * 60}")
    print(f"Phase 4d Results:")
    print(f"  Initial:     {results['initial_latent_dist']:.2f}")
    print(f"  Final:       {results['final_latent_dist']:.2f}")
    print(f"  Min:         {results['min_latent_dist']:.2f}")
    print(f"  Improvement: {results['improvement_pct']:.1f}%")
    print(f"  Env Reward:  {results['total_env_reward']:.2f}")
    print(f"{'=' * 60}")

    imageio.mimwrite("/results/mpc_cem_4d_rollout.mp4", frames, fps=30)
    with open("/results/phase4d_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(latent_distances, color='#4361ee', linewidth=1.5)
    axes[0].axhline(y=init_latent_dist, color='red', linestyle='--', alpha=0.5, label='Initial')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Latent Distance to Goal")
    axes[0].set_title("Phase 4d: ResBlock Dynamics (4.2M params)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.cumsum(rewards_log), color='#f72585', linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative Env Reward")
    axes[1].set_title("Phase 4d: Cumulative Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/results/mpc_cem_4d_metrics.png", dpi=150)

    np.save("/results/latent_distances_4d.npy", np.array(latent_distances))
    results_vol.commit()
    return results


@app.local_entrypoint()
def main():
    import subprocess

    print("=" * 70)
    print("Phase 4d: ResBlock Dynamics + CEM Evaluation")
    print("=" * 70)

    print("\n[1] Training ResBlock dynamics model...")
    train_result = train_resblock.remote()
    print(f"\nTraining: {train_result}")

    print("\n[2] Running CEM evaluation...")
    eval_result = run_cem_eval.remote()
    print(f"\nEvaluation: {eval_result}")

    # Download
    out_dir = Path("train_robots/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "dynamics_resblock.pt",
        "dynamics_resblock_loss.png",
        "mpc_cem_4d_rollout.mp4",
        "mpc_cem_4d_metrics.png",
        "phase4d_results.json",
        "latent_distances_4d.npy",
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

    print("\n" + "=" * 70)
    print("Phase Comparison")
    print("=" * 70)
    print(f"  4b (MLP 1.2M, 1-step):     41.3% improvement, 6.0 env reward")
    print(f"  4c (MLP 1.2M, multi-step):  31.4% improvement, 0.0 env reward")
    print(f"  4d (ResBlock 4.2M, 1-step): {eval_result['improvement_pct']:.1f}% improvement, "
          f"{eval_result['total_env_reward']:.1f} env reward")
