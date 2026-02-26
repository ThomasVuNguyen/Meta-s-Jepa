"""
Phase 4e — Retrain dynamics models on 1M transitions (5000 episodes)
====================================================================
Single Modal script with 3 stages:
  Stage 1: Train MLP (1.2M params) — 100 epochs
  Stage 2: Train ResBlock (10.5M params, +dropout) — 150 epochs
  Stage 3: CEM evaluation of both models

Data: reacher_easy_5k.npz from Modal volume (1M transitions, 8.2 GB)
Estimated: ~55 min, ~$1.10 on A10G
"""

import modal

app = modal.App("vjepa2-phase4e")

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
        "torch", "torchvision", "numpy", "Pillow",
        "matplotlib", "imageio", "imageio-ffmpeg",
    ])
    .env({
        "MUJOCO_GL": "osmesa",
        "TRANSFORMERS_CACHE": "/cache/hf",
    })
)

HF_REPO = "ThomasTheMaker/vjepa2-reacher-world-model"


# ── Model definitions ───────────────────────────────────────────────

def make_mlp_model():
    import torch.nn as nn

    class DynamicsPredictor(nn.Module):
        """MLP dynamics: 1.2M params. Same as Phase 4b."""
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
            import torch
            x = torch.cat([z_t, a_t], dim=-1)
            return z_t + self.net(x)

    return DynamicsPredictor


def make_resblock_model():
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x + self.net(x))

    class DynamicsPredictor(nn.Module):
        """ResBlock dynamics: 10.5M params + dropout(0.1)."""
        def __init__(self, latent_dim=1024, action_dim=2, hidden_dim=1024, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(
                ResBlock(hidden_dim, dropout),
                ResBlock(hidden_dim, dropout),
                ResBlock(hidden_dim, dropout),
                ResBlock(hidden_dim, dropout),
            )
            self.output_proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, z_t, a_t):
            import torch
            x = torch.cat([z_t, a_t], dim=-1)
            h = self.input_proj(x)
            h = self.blocks(h)
            return z_t + self.output_proj(h)

    return DynamicsPredictor


# ── Stage 1: Train MLP ─────────────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_mlp():
    """Train MLP dynamics on 1M transitions."""
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    BATCH_SIZE = 2048
    EPOCHS = 30
    LR = 3e-4

    print("Loading 5k dataset...")
    data = np.load("/demos/reacher_easy_5k.npz")
    z_t    = torch.tensor(data['z_t'], dtype=torch.float32)
    a_t    = torch.tensor(data['a_t'], dtype=torch.float32)
    z_next = torch.tensor(data['z_next'], dtype=torch.float32)
    print(f"  z_t: {z_t.shape} | a_t: {a_t.shape}")

    n = len(z_t)
    train_size = int(0.9 * n)  # 900k train, 100k val

    train_ds = TensorDataset(z_t[:train_size], a_t[:train_size], z_next[:train_size])
    val_ds   = TensorDataset(z_t[train_size:], a_t[train_size:], z_next[train_size:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    Model = make_mlp_model()
    model = Model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Stage 1: MLP Dynamics | {n_params:,} params")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for z, a, zn in train_loader:
            z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(z, a), zn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * z.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, zn in val_loader:
                z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
                val_loss += criterion(model(z, a), zn).item() * z.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/results/dynamics_mlp_5k.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {epoch_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nMLP done. Best val: {best_val_loss:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Smooth L1 Loss")
    plt.title(f"Phase 4e — MLP ({n_params/1e6:.1f}M) on 1M transitions")
    plt.legend(); plt.grid(True)
    plt.savefig("/results/loss_4e_mlp.png", dpi=150)

    results_vol.commit()
    return {"model": "mlp", "params": n_params, "best_val_loss": float(best_val_loss)}


# ── Stage 2: Train ResBlock ─────────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_resblock():
    """Train ResBlock dynamics on 1M transitions (with dropout)."""
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    BATCH_SIZE = 2048
    EPOCHS = 50
    LR = 3e-4

    print("Loading 5k dataset...")
    data = np.load("/demos/reacher_easy_5k.npz")
    z_t    = torch.tensor(data['z_t'], dtype=torch.float32)
    a_t    = torch.tensor(data['a_t'], dtype=torch.float32)
    z_next = torch.tensor(data['z_next'], dtype=torch.float32)

    n = len(z_t)
    train_size = int(0.9 * n)

    train_ds = TensorDataset(z_t[:train_size], a_t[:train_size], z_next[:train_size])
    val_ds   = TensorDataset(z_t[train_size:], a_t[train_size:], z_next[train_size:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    Model = make_resblock_model()
    model = Model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Stage 2: ResBlock Dynamics | {n_params:,} params | +dropout(0.1)")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for z, a, zn in train_loader:
            z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(z, a), zn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * z.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, zn in val_loader:
                z, a, zn = z.to(DEVICE), a.to(DEVICE), zn.to(DEVICE)
                val_loss += criterion(model(z, a), zn).item() * z.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/results/dynamics_resblock_5k.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {epoch_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nResBlock done. Best val: {best_val_loss:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Smooth L1 Loss")
    plt.title(f"Phase 4e — ResBlock ({n_params/1e6:.1f}M +dropout) on 1M transitions")
    plt.legend(); plt.grid(True)
    plt.savefig("/results/loss_4e_resblock.png", dpi=150)

    results_vol.commit()
    return {"model": "resblock", "params": n_params, "best_val_loss": float(best_val_loss)}


# ── Stage 3: CEM Evaluation ────────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_cem_eval():
    """Evaluate both dynamics models with CEM planner."""
    import torch, numpy as np, os, json
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    from transformers import AutoModel
    from PIL import Image
    import torchvision.transforms as T
    import imageio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from huggingface_hub import HfApi

    DEVICE = "cuda"

    vjepa_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    @torch.no_grad()
    def encode_image(model, img):
        x = vjepa_transform(img)
        clips_t = torch.stack([x]*8).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        out = model(pixel_values_videos=clips_t, return_dict=True)
        return out.last_hidden_state.mean(dim=1)

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
            mean = torch.zeros(self.horizon, self.action_dim, device=DEVICE)
            if self._prev_mean is not None:
                mean[:-1] = self._prev_mean[1:]
            std = torch.ones(self.horizon, self.action_dim, device=DEVICE) * 0.5

            best_actions = None
            best_cost = float('inf')

            for _ in range(self.num_iterations):
                noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=DEVICE)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1, 1)

                # Evaluate
                z_curr = z_t.expand(self.num_samples, -1)
                z_goal_exp = z_goal.expand(self.num_samples, -1)
                total_cost = torch.zeros(self.num_samples, device=DEVICE)
                for t in range(self.horizon):
                    z_curr = self.dynamics_model(z_curr, actions[:, t, :])
                    weight = (t + 1) / self.horizon
                    total_cost += weight * torch.norm(z_curr - z_goal_exp, dim=-1)

                elite_idx = torch.topk(total_cost, self.num_elites, largest=False).indices
                elite_actions = actions[elite_idx]
                if total_cost[elite_idx[0]].item() < best_cost:
                    best_cost = total_cost[elite_idx[0]].item()
                    best_actions = actions[elite_idx[0]]
                mean = elite_actions.mean(dim=0)
                std = elite_actions.std(dim=0).clamp(min=0.05)

            self._prev_mean = mean.clone()
            return best_actions[0].cpu().numpy()

    # --- Load V-JEPA ---
    print("Loading V-JEPA 2...")
    vjepa_model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True,
        cache_dir="/cache/hf"
    ).to(DEVICE).eval()

    # --- Generate expert goal ---
    print("Generating expert goal...")
    raw_env = suite.load("reacher", "easy", task_kwargs={'random': 42})
    ts = raw_env.reset()
    obs = ts.observation
    best_dist, best_frame = float('inf'), None
    for _ in range(300):
        action = np.clip(obs["to_target"] * 5.0, -1, 1).astype(np.float32)
        ts = raw_env.step(action)
        obs = ts.observation
        dist = np.linalg.norm(obs["to_target"])
        if dist < best_dist:
            best_dist = dist
            best_frame = raw_env.physics.render(height=224, width=224, camera_id=0).copy()
    z_goal = encode_image(vjepa_model, Image.fromarray(best_frame))
    print(f"  Goal dist: {best_dist:.4f}")

    # --- Eval both models ---
    models_to_eval = [
        ("MLP (1.2M)", make_mlp_model(), "/results/dynamics_mlp_5k.pt", "4e_mlp"),
        ("ResBlock (10.5M)", make_resblock_model(), "/results/dynamics_resblock_5k.pt", "4e_resblock"),
    ]

    all_results = []
    for model_name, ModelClass, ckpt_path, tag in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        dynamics_model = ModelClass().to(DEVICE)
        dynamics_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        dynamics_model.eval()

        planner = CEMPlanner(dynamics_model, num_samples=500, horizon=10,
                             num_iterations=5, elite_frac=0.1)

        env = suite.load("reacher", "easy", task_kwargs={'random': 42})
        env = pixels.Wrapper(env, pixels_only=False, render_kwargs={
            'height': 224, 'width': 224, 'camera_id': 0
        })

        time_step = env.reset()
        init_img = Image.fromarray(time_step.observation['pixels'])
        z_init = encode_image(vjepa_model, init_img)
        init_latent_dist = torch.norm(z_init - z_goal).item()

        frames, latent_distances, rewards_log = [], [], []
        total_reward = 0.0

        for step in range(200):
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

            if (step + 1) % 50 == 0:
                print(f"  Step {step+1:3d} | LatDist: {lat_dist:.2f} | Reward: {reward:.1f}")

        final_latent_dist = latent_distances[-1]
        min_latent_dist = min(latent_distances)
        improvement = (1 - final_latent_dist / init_latent_dist) * 100

        result = {
            "model": model_name,
            "tag": tag,
            "init_latent_dist": init_latent_dist,
            "final_latent_dist": final_latent_dist,
            "min_latent_dist": min_latent_dist,
            "improvement_pct": improvement,
            "total_env_reward": total_reward,
        }
        all_results.append(result)

        print(f"\n  Init: {init_latent_dist:.2f} → Final: {final_latent_dist:.2f} "
              f"(min: {min_latent_dist:.2f})")
        print(f"  Improvement: {improvement:.1f}% | Reward: {total_reward:.1f}")

        # Save video
        imageio.mimsave(f"/results/rollout_4e_{tag}.mp4", frames, fps=20)

        # Save latent distance plot
        plt.figure(figsize=(10, 5))
        plt.plot(latent_distances, label=f"Phase 4e {model_name}")
        plt.axhline(y=init_latent_dist, color='r', linestyle='--', alpha=0.5, label="Initial")
        plt.xlabel("Step"); plt.ylabel("Latent Distance to Goal")
        plt.title(f"Phase 4e — {model_name} on 1M transitions")
        plt.legend(); plt.grid(True)
        plt.savefig(f"/results/latent_dist_4e_{tag}.png", dpi=150)
        plt.close()

        # Save latent distances for later analysis
        np.save(f"/results/latent_distances_{tag}.npy", np.array(latent_distances))

        # Reset planner for next model
        planner._prev_mean = None

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("Phase 4e Results Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Improvement':>12} {'Min Dist':>10} {'Reward':>8}")
    print("-" * 60)
    print(f"{'Phase 4b baseline':<25} {'41.3%':>12} {'7.80':>10} {'6.0':>8}")
    for r in all_results:
        print(f"{r['model']:<25} {r['improvement_pct']:>11.1f}% "
              f"{r['min_latent_dist']:>10.2f} {r['total_env_reward']:>8.1f}")

    # Save results JSON
    with open("/results/phase4e_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Upload to HuggingFace
    print("\nUploading to HuggingFace...")
    api = HfApi(token=os.environ["HF_TOKEN"])
    upload_files = [
        ("/results/dynamics_mlp_5k.pt", "models/dynamics_mlp_1.2M_5k.pt"),
        ("/results/dynamics_resblock_5k.pt", "models/dynamics_resblock_10.5M_5k.pt"),
        ("/results/loss_4e_mlp.png", "results/loss_4e_mlp.png"),
        ("/results/loss_4e_resblock.png", "results/loss_4e_resblock.png"),
        ("/results/latent_dist_4e_4e_mlp.png", "results/latent_dist_4e_mlp.png"),
        ("/results/latent_dist_4e_4e_resblock.png", "results/latent_dist_4e_resblock.png"),
        ("/results/phase4e_results.json", "results/phase4e_results.json"),
    ]
    for local, remote in upload_files:
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=HF_REPO, repo_type="dataset")
            print(f"  ✅ {remote}")

    results_vol.commit()
    return all_results


# ── Entrypoint ──────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Phase 4e: Retrain on 1M transitions + CEM Evaluation")
    print("=" * 70)

    # Stages 1 & 2: Both models already trained (checkpoints on volume)
    print("\n[Stage 1] MLP — SKIPPED (dynamics_mlp_5k.pt exists)")
    print("[Stage 2] ResBlock — SKIPPED (dynamics_resblock_5k.pt exists)")

    # Stage 3: CEM Evaluation
    print("\n[Stage 3] Running CEM evaluation...")
    eval_results = run_cem_eval.remote()

    print("\n" + "=" * 70)
    print("Phase 4e Complete!")
    print("=" * 70)
    for r in eval_results:
        print(f"  {r['model']}: {r['improvement_pct']:.1f}% improvement, "
              f"reward={r['total_env_reward']:.1f}")
