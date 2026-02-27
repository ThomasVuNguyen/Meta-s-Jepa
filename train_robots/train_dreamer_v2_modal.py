"""
Phase 5b — Reward Model + Hybrid Dreamer Actor-Critic
=====================================================
Fix Phase 5's zero-reward problem with three improvements:
  1. Train a reward predictor R(z_t, a_t) → reward from dataset
  2. Hybrid training reward: α·R_pred + β·(-latent_dist) + γ·||a||
  3. Action magnitude bonus to prevent tiny actions

Stages:
  1. Train reward model (~5 min)
  2. Train improved Dreamer actor-critic (~25 min)
  3. Evaluate in real environment (~15 min)

Cost: ~$0.90 on A10G
"""

import modal

app = modal.App("vjepa2-phase5b-dreamer-v2")

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


# ── Model definitions ──────────────────────────────────────────────

def make_dynamics_model():
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
            import torch
            x = torch.cat([z_t, a_t], dim=-1)
            return z_t + self.net(x)

    return DynamicsPredictor


def make_reward_model():
    import torch.nn as nn

    class RewardPredictor(nn.Module):
        """
        Predict environment reward from (z_t, a_t).
        R(z_t, a_t) → scalar reward ∈ [0, 1]
        ~270K params
        """
        def __init__(self, latent_dim=1024, action_dim=2, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # reward is in [0, 1]
            )

        def forward(self, z_t, a_t):
            import torch
            x = torch.cat([z_t, a_t], dim=-1)
            return self.net(x).squeeze(-1)

    return RewardPredictor


def make_actor():
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, latent_dim=1024, action_dim=2, hidden_dim=256):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

        def forward(self, z):
            h = self.trunk(z)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-5, 2)
            return mean, log_std

        def sample(self, z):
            mean, log_std = self.forward(z)
            std = log_std.exp()
            noise = torch.randn_like(mean)
            raw_action = mean + std * noise
            action = torch.tanh(raw_action)

            log_prob = (-0.5 * (noise ** 2) - log_std - 0.5 * torch.log(
                torch.tensor(2 * 3.14159265, device=z.device))).sum(-1)
            log_prob -= (2 * (torch.log(torch.tensor(2.0, device=z.device)) -
                        raw_action -
                        torch.nn.functional.softplus(-2 * raw_action))).sum(-1)
            return action, log_prob

        def act_deterministic(self, z):
            mean, _ = self.forward(z)
            return torch.tanh(mean)

    return Actor


def make_critic():
    import torch.nn as nn

    class Critic(nn.Module):
        def __init__(self, latent_dim=1024, hidden_dim=256):
            super().__init__()
            self.v1 = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.v2 = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z):
            return self.v1(z).squeeze(-1), self.v2(z).squeeze(-1)

    return Critic


# ── Stage 1: Train Reward Model ────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_reward_model():
    """Train R(z_t, a_t) → reward from dataset."""
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    BATCH_SIZE = 2048
    EPOCHS = 50
    LR = 3e-4

    print("Loading dataset...")
    data = np.load("/demos/reacher_easy_5k.npz")
    z_t = torch.tensor(data['z_t'], dtype=torch.float32)
    a_t = torch.tensor(data['a_t'], dtype=torch.float32)
    rewards = torch.tensor(data['rewards'], dtype=torch.float32)

    n = len(z_t)
    print(f"  {n:,} transitions")
    print(f"  Rewards — min: {rewards.min():.4f}, max: {rewards.max():.4f}, "
          f"mean: {rewards.mean():.4f}, non-zero: {(rewards > 0).sum()}/{n} "
          f"({(rewards > 0).float().mean()*100:.1f}%)")

    # Train/val split
    train_size = int(0.9 * n)
    train_ds = TensorDataset(z_t[:train_size], a_t[:train_size], rewards[:train_size])
    val_ds = TensorDataset(z_t[train_size:], a_t[train_size:], rewards[train_size:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    RewardPredictor = make_reward_model()
    model = RewardPredictor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Use weighted BCE since rewards are very sparse
    pos_weight = ((rewards == 0).sum() / max((rewards > 0).sum(), 1)).item()
    pos_weight = min(pos_weight, 50.0)  # cap at 50x
    print(f"  Pos weight: {pos_weight:.1f}x")

    # Use MSE since rewards can be continuous [0, 1]
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Stage 1: Reward Model | {n_params:,} params")
    print(f"{'='*60}\n")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for z, a, r in train_loader:
            z, a, r = z.to(DEVICE), a.to(DEVICE), r.to(DEVICE)
            optimizer.zero_grad()
            r_pred = model(z, a)
            loss = criterion(r_pred, r)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * z.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, r in val_loader:
                z, a, r = z.to(DEVICE), a.to(DEVICE), r.to(DEVICE)
                val_loss += criterion(model(z, a), r).item() * z.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/results/reward_model.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {epoch_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    # Test: check reward predictions on high/low reward states
    model.eval()
    with torch.no_grad():
        # High reward states
        high_idx = torch.topk(rewards, min(100, len(rewards))).indices
        high_pred = model(z_t[high_idx].to(DEVICE), a_t[high_idx].to(DEVICE))
        print(f"\nHigh-reward states: true={rewards[high_idx].mean():.4f}, "
              f"pred={high_pred.mean():.4f}")

        # Zero reward states
        zero_idx = (rewards == 0).nonzero()[:100].squeeze()
        zero_pred = model(z_t[zero_idx].to(DEVICE), a_t[zero_idx].to(DEVICE))
        print(f"Zero-reward states: true={rewards[zero_idx].mean():.4f}, "
              f"pred={zero_pred.mean():.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"Phase 5b — Reward Model ({n_params/1e3:.0f}K params)")
    plt.legend(); plt.grid(True)
    plt.savefig("/results/loss_reward_model.png", dpi=150)

    results_vol.commit()
    return {"params": n_params, "best_val_loss": float(best_val_loss)}


# ── Stage 2: Hybrid Dreamer ────────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_dreamer_v2():
    """Train actor-critic with hybrid reward: R_pred + latent_dist + action_mag."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    HORIZON = 15
    BATCH_SIZE = 128       # larger batch for more stable gradients
    EPOCHS = 500
    GAMMA = 0.99
    LAMBDA = 0.95
    ACTOR_LR = 1e-4        # lower LR for more stability
    CRITIC_LR = 3e-4
    ENTROPY_COEFF = 0.01
    TARGET_TAU = 0.02

    # Reward weights
    REWARD_WEIGHT = 10.0    # weight for predicted reward (most important!)
    DIST_WEIGHT = 0.1       # small weight for latent distance (secondary signal)
    ACTION_MAG_WEIGHT = 0.5 # bonus for larger actions (prevents tiny actions)

    # --- Load frozen models ---
    print("Loading frozen dynamics model...")
    DynamicsPredictor = make_dynamics_model()
    dynamics = DynamicsPredictor().to(DEVICE)
    dynamics.load_state_dict(torch.load("/results/dynamics_mlp_5k.pt", weights_only=True))
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad = False

    print("Loading frozen reward model...")
    RewardPredictor = make_reward_model()
    reward_model = RewardPredictor().to(DEVICE)
    reward_model.load_state_dict(torch.load("/results/reward_model.pt", weights_only=True))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # --- Load starting states ---
    print("Loading dataset...")
    data = np.load("/demos/reacher_easy_5k.npz")
    z_all = torch.tensor(data['z_t'], dtype=torch.float32).to(DEVICE)
    n_states = len(z_all)

    # Goal: mean of highest-reward states
    rewards_data = torch.tensor(data['rewards'], dtype=torch.float32)
    if rewards_data.sum() > 0:
        top_idx = torch.topk(rewards_data, min(500, len(rewards_data))).indices
        z_goal = z_all[top_idx].mean(dim=0, keepdim=True)
    else:
        ep_len = 200
        n_episodes = n_states // ep_len
        goal_idx = []
        for ep in range(n_episodes):
            for t in range(190, 200):
                goal_idx.append(ep * ep_len + t)
        z_goal = z_all[goal_idx].mean(dim=0, keepdim=True)

    z_goal_norm = z_goal.norm()

    # --- Create trainable models ---
    Actor = make_actor()
    Critic = make_critic()

    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    critic_target = Critic().to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    print(f"\n{'='*60}")
    print(f"Phase 5b: Hybrid Dreamer (Reward Model + Action Mag)")
    print(f"  Reward weight: {REWARD_WEIGHT} | Dist weight: {DIST_WEIGHT} | "
          f"Action mag: {ACTION_MAG_WEIGHT}")
    print(f"  Horizon: {HORIZON} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print(f"{'='*60}\n")

    # --- Training loop ---
    actor_losses_log, critic_losses_log = [], []
    reward_pred_log, reward_total_log, action_mag_log = [], [], []
    best_avg_reward = -float('inf')

    for epoch in range(EPOCHS):
        # Sample starting states
        idx = torch.randint(0, n_states, (BATCH_SIZE,), device=DEVICE)
        z_start = z_all[idx]

        # ── Imagine forward ──
        imagined_rewards = []
        imagined_log_probs = []
        imagined_z = [z_start]

        z_t = z_start
        for h in range(HORIZON):
            action, log_prob = actor.sample(z_t)
            z_next = dynamics(z_t, action)

            # Hybrid reward:
            # 1. Predicted env reward (most important)
            r_pred = reward_model(z_t, action)

            # 2. Negative latent distance (secondary)
            dist_to_goal = torch.norm(z_next - z_goal.expand(BATCH_SIZE, -1), dim=-1)
            r_dist = -dist_to_goal / z_goal_norm

            # 3. Action magnitude bonus (prevents tiny actions)
            action_mag = torch.norm(action, dim=-1)
            r_action = action_mag  # reward larger actions

            # Combined
            reward = (REWARD_WEIGHT * r_pred +
                      DIST_WEIGHT * r_dist +
                      ACTION_MAG_WEIGHT * r_action)

            imagined_rewards.append(reward)
            imagined_log_probs.append(log_prob)
            imagined_z.append(z_next)

            z_t = z_next

        imagined_z = torch.stack(imagined_z)              # [H+1, B, 1024]
        imagined_rewards = torch.stack(imagined_rewards)    # [H, B]
        imagined_log_probs = torch.stack(imagined_log_probs)

        # ── TD-λ targets ──
        with torch.no_grad():
            v1_t, v2_t = critic_target(imagined_z[1:].reshape(-1, 1024))
            v_target = torch.min(v1_t, v2_t).reshape(HORIZON, BATCH_SIZE)

            v1_f, v2_f = critic_target(imagined_z[-1])
            v_bootstrap = torch.min(v1_f, v2_f)

            returns = torch.zeros(HORIZON, BATCH_SIZE, device=DEVICE)
            last_return = v_bootstrap
            for h in reversed(range(HORIZON)):
                td_target = imagined_rewards[h] + GAMMA * v_target[h]
                last_return = (1 - LAMBDA) * td_target + LAMBDA * (
                    imagined_rewards[h] + GAMMA * last_return)
                returns[h] = last_return

        # ── Train critic ──
        v1_pred, v2_pred = critic(imagined_z[:-1].reshape(-1, 1024))
        v1_pred = v1_pred.reshape(HORIZON, BATCH_SIZE)
        v2_pred = v2_pred.reshape(HORIZON, BATCH_SIZE)

        critic_loss = (nn.functional.mse_loss(v1_pred, returns) +
                       nn.functional.mse_loss(v2_pred, returns))

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        critic_opt.step()

        # ── Train actor ──
        z_t = z_start.detach()
        actor_reward_sum = torch.zeros(BATCH_SIZE, device=DEVICE)

        for h in range(HORIZON):
            action, log_prob_h = actor.sample(z_t)
            z_next = dynamics(z_t, action)

            with torch.no_grad():
                v1, v2 = critic(z_next)
                value = torch.min(v1, v2)

            # Same hybrid reward
            r_pred = reward_model(z_t, action)
            dist_to_goal = torch.norm(z_next - z_goal.expand(BATCH_SIZE, -1), dim=-1)
            r_dist = -dist_to_goal / z_goal_norm
            action_mag = torch.norm(action, dim=-1)

            reward = (REWARD_WEIGHT * r_pred +
                      DIST_WEIGHT * r_dist +
                      ACTION_MAG_WEIGHT * action_mag)

            weight = GAMMA ** h
            actor_reward_sum += weight * (reward + GAMMA * value +
                                          ENTROPY_COEFF * (-log_prob_h))
            z_t = z_next

        actor_loss = -actor_reward_sum.mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        actor_opt.step()

        # ── Soft target update ──
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.lerp_(p.data, TARGET_TAU)

        # ── Logging ──
        avg_total_reward = imagined_rewards.mean().item()
        actor_losses_log.append(actor_loss.item())
        critic_losses_log.append(critic_loss.item())
        reward_total_log.append(avg_total_reward)

        # Track action magnitudes
        with torch.no_grad():
            test_actions, _ = actor.sample(z_start[:16])
            avg_action_mag = test_actions.abs().mean().item()
        action_mag_log.append(avg_action_mag)

        if avg_total_reward > best_avg_reward:
            best_avg_reward = avg_total_reward
            torch.save(actor.state_dict(), "/results/actor_dreamer_v2.pt")
            torch.save(critic.state_dict(), "/results/critic_dreamer_v2.pt")

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
                  f"Reward: {avg_total_reward:.4f} | "
                  f"Actor L: {actor_loss.item():.3f} | "
                  f"Critic L: {critic_loss.item():.4f} | "
                  f"Action |a|: {avg_action_mag:.3f} | "
                  f"Best: {best_avg_reward:.4f}")

    print(f"\nTraining done. Best reward: {best_avg_reward:.4f}")
    print(f"Final avg |action|: {action_mag_log[-1]:.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(reward_total_log)
    axes[0, 0].set_title("Avg Hybrid Reward"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True)

    axes[0, 1].plot(action_mag_log)
    axes[0, 1].set_title("Avg |Action| Magnitude"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label="Min desirable")
    axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(actor_losses_log)
    axes[1, 0].set_title("Actor Loss"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)

    axes[1, 1].plot(critic_losses_log)
    axes[1, 1].set_title("Critic Loss"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True)

    plt.suptitle("Phase 5b: Hybrid Dreamer Training")
    plt.tight_layout()
    plt.savefig("/results/dreamer_v2_training.png", dpi=150)

    results_vol.commit()
    return {
        "best_reward": best_avg_reward,
        "final_action_mag": action_mag_log[-1],
    }


# ── Stage 3: Evaluate ──────────────────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def evaluate_actor_v2():
    """Evaluate the improved actor in dm_control."""
    import torch, numpy as np, os, json
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    from transformers import AutoModel
    from PIL import Image
    import torchvision.transforms as T
    import imageio
    import matplotlib; matplotlib.use("Agg")
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
        clips_t = torch.stack([x] * 8).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        out = model(pixel_values_videos=clips_t, return_dict=True)
        return out.last_hidden_state.mean(dim=1).float()

    # --- Load V-JEPA ---
    print("Loading V-JEPA 2...")
    vjepa_model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True,
        cache_dir="/cache/hf"
    ).to(DEVICE).eval()

    # --- Load actor ---
    print("Loading trained actor v2...")
    Actor = make_actor()
    actor = Actor().to(DEVICE)
    actor.load_state_dict(torch.load("/results/actor_dreamer_v2.pt", weights_only=True))
    actor.eval()

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

    # --- Run evaluation ---
    env = suite.load("reacher", "easy", task_kwargs={'random': 42})
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs={
        'height': 224, 'width': 224, 'camera_id': 0
    })

    print(f"\n{'='*60}")
    print("Phase 5b: Hybrid Dreamer Evaluation")
    print(f"{'='*60}")

    time_step = env.reset()
    z_init = encode_image(vjepa_model, Image.fromarray(time_step.observation['pixels']))
    init_latent_dist = torch.norm(z_init - z_goal).item()

    frames, latent_distances, rewards_log = [], [], []
    total_reward, action_mags = 0.0, []

    for step in range(200):
        img_t = Image.fromarray(time_step.observation['pixels'])
        frames.append(np.array(img_t))
        z_t = encode_image(vjepa_model, img_t)
        lat_dist = torch.norm(z_t - z_goal).item()
        latent_distances.append(lat_dist)

        with torch.no_grad():
            action = actor.act_deterministic(z_t).cpu().numpy().flatten()

        action_mags.append(np.linalg.norm(action))
        time_step = env.step(action)
        reward = time_step.reward or 0.0
        total_reward += reward
        rewards_log.append(reward)

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:3d} | LatDist: {lat_dist:.2f} | "
                  f"Reward: {reward:.3f} | |a|: {np.linalg.norm(action):.3f} | "
                  f"Action: [{action[0]:.2f}, {action[1]:.2f}]")

    final_latent_dist = latent_distances[-1]
    min_latent_dist = min(latent_distances)
    improvement = (1 - final_latent_dist / init_latent_dist) * 100

    result = {
        "phase": "5b",
        "model": "Hybrid Dreamer v2",
        "init_latent_dist": init_latent_dist,
        "final_latent_dist": final_latent_dist,
        "min_latent_dist": min_latent_dist,
        "improvement_pct": improvement,
        "total_env_reward": total_reward,
        "max_step_reward": max(rewards_log),
        "avg_action_magnitude": float(np.mean(action_mags)),
    }

    # Comparison table
    print(f"\n{'='*60}")
    print("Full Comparison")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Improvement':>12} {'Reward':>8} {'|a| avg':>8}")
    print("-" * 63)
    print(f"{'Phase 4e MLP CEM':<30} {'25.5%':>12} {'29.0':>8} {'N/A':>8}")
    print(f"{'Phase 4e ResBlock CEM':<30} {'44.1%':>12} {'0.0':>8} {'N/A':>8}")
    print(f"{'Phase 5 Dreamer v1':<30} {'37.5%':>12} {'0.0':>8} {'0.05':>8}")
    print(f"{'Phase 5b Hybrid Dreamer':<30} {improvement:>11.1f}% "
          f"{total_reward:>8.1f} {np.mean(action_mags):>8.3f}")

    # Save artifacts
    imageio.mimsave("/results/rollout_5b_dreamer.mp4", frames, fps=20)

    plt.figure(figsize=(10, 5))
    plt.plot(latent_distances, label="Phase 5b Hybrid Dreamer", linewidth=2)
    plt.axhline(y=init_latent_dist, color='r', linestyle='--', alpha=0.5, label="Initial")
    plt.xlabel("Step"); plt.ylabel("Latent Distance to Goal")
    plt.title("Phase 5b — Hybrid Dreamer Evaluation")
    plt.legend(); plt.grid(True)
    plt.savefig("/results/latent_dist_5b.png", dpi=150)

    np.save("/results/latent_distances_5b.npy", np.array(latent_distances))

    with open("/results/phase5b_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Upload to HuggingFace
    print("\nUploading to HuggingFace...")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    upload_files = [
        ("/results/actor_dreamer_v2.pt", "models/actor_dreamer_v2.pt"),
        ("/results/reward_model.pt", "models/reward_model.pt"),
        ("/results/dreamer_v2_training.png", "results/dreamer_v2_training.png"),
        ("/results/latent_dist_5b.png", "results/latent_dist_5b.png"),
        ("/results/phase5b_results.json", "results/phase5b_results.json"),
    ]
    for local, remote in upload_files:
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id="ThomasTheMaker/vjepa2-reacher-world-model",
                            repo_type="dataset")
            print(f"  ✅ {remote}")

    results_vol.commit()
    return result


# ── Entrypoint ──────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Phase 5b: Reward Model + Hybrid Dreamer Actor-Critic")
    print("=" * 70)

    # Stage 1: Reward model
    print("\n[Stage 1] Training reward model R(z_t, a_t)...")
    rm_result = train_reward_model.remote()
    print(f"  Result: {rm_result}")

    # Stage 2: Hybrid Dreamer
    print("\n[Stage 2] Training hybrid Dreamer actor-critic...")
    dreamer_result = train_dreamer_v2.remote()
    print(f"  Result: {dreamer_result}")

    # Stage 3: Evaluate
    print("\n[Stage 3] Evaluating in real environment...")
    eval_result = evaluate_actor_v2.remote()

    print("\n" + "=" * 70)
    print("Phase 5b Complete!")
    print("=" * 70)
    print(f"  Improvement: {eval_result['improvement_pct']:.1f}%")
    print(f"  Env Reward: {eval_result['total_env_reward']:.1f}")
    print(f"  Avg |action|: {eval_result['avg_action_magnitude']:.3f}")
    print(f"  (Baseline: Phase 4e MLP CEM = 29.0 reward)")
