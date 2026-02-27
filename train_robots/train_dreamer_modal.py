"""
Phase 5 — Dreamer-Style Latent Actor-Critic
============================================
Train actor π(z_t) → a_t and critic V(z_t) by "dreaming" inside the
frozen MLP dynamics model. Then evaluate the actor in dm_control.

Algorithm:
  1. Sample real z_0 from dataset
  2. Dream forward H steps: a_t = π(z_t), z_{t+1} = f(z_t, a_t)
  3. Reward = -||z_t - z_goal|| at each dream step
  4. Train critic on TD-λ targets from imagined rewards
  5. Train actor to maximize critic values via backprop through dream

Cost: ~$0.70 on A10G (~35 min total)
"""

import modal

app = modal.App("vjepa2-phase5-dreamer")

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


# ── Dynamics model (frozen, from Phase 4e) ──────────────────────────

def make_dynamics_model():
    import torch.nn as nn

    class DynamicsPredictor(nn.Module):
        """MLP dynamics: 1.2M params. Frozen during Phase 5."""
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


# ── Actor and Critic ────────────────────────────────────────────────

def make_actor():
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        """
        Stochastic actor: outputs mean and log_std for Gaussian policy.
        π(z_t) → (mean, log_std) → sample → tanh squash → a_t ∈ [-1,1]
        ~265K params
        """
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

            # Log prob with tanh correction
            log_prob = (-0.5 * (noise ** 2) - log_std - 0.5 * torch.log(
                torch.tensor(2 * 3.14159265))).sum(-1)
            log_prob -= (2 * (torch.log(torch.tensor(2.0)) - raw_action -
                        torch.nn.functional.softplus(-2 * raw_action))).sum(-1)
            return action, log_prob

        def act_deterministic(self, z):
            mean, _ = self.forward(z)
            return torch.tanh(mean)

    return Actor


def make_critic():
    import torch.nn as nn

    class Critic(nn.Module):
        """
        Value function V(z_t) → scalar value.
        Two independent heads for double-Q style stability.
        ~265K params per head, ~530K total
        """
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


# ── Stage 1: Train in imagination ──────────────────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
)
def train_dreamer():
    """Train actor-critic by dreaming inside the frozen dynamics model."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dm_control import suite

    DEVICE = "cuda"
    HORIZON = 15          # imagination rollout length
    BATCH_SIZE = 64       # number of parallel dreams
    EPOCHS = 500          # training epochs
    GAMMA = 0.99          # discount factor
    LAMBDA = 0.95         # TD-λ
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ENTROPY_COEFF = 0.01  # SAC-style entropy bonus
    TARGET_TAU = 0.02     # soft target update rate

    # --- Load frozen dynamics model ---
    print("Loading frozen MLP dynamics model...")
    DynamicsPredictor = make_dynamics_model()
    dynamics = DynamicsPredictor().to(DEVICE)
    dynamics.load_state_dict(torch.load("/results/dynamics_mlp_5k.pt", weights_only=True))
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad = False
    print(f"  Dynamics params: {sum(p.numel() for p in dynamics.parameters()):,} (frozen)")

    # --- Load dataset for starting states ---
    print("Loading dataset for real starting states...")
    data = np.load("/demos/reacher_easy_5k.npz")
    z_all = torch.tensor(data['z_t'], dtype=torch.float32).to(DEVICE)
    n_states = len(z_all)
    print(f"  {n_states:,} starting states available")

    # --- Generate expert goal ---
    print("Generating expert goal state...")
    raw_env = suite.load("reacher", "easy", task_kwargs={'random': 42})
    ts = raw_env.reset()
    obs = ts.observation
    best_dist, best_obs = float('inf'), None

    for _ in range(300):
        action = np.clip(obs["to_target"] * 5.0, -1, 1).astype(np.float32)
        ts = raw_env.step(action)
        obs = ts.observation
        dist = np.linalg.norm(obs["to_target"])
        if dist < best_dist:
            best_dist = dist
            # We don't have VJEPA here, so use the dataset's goal states
            # Pick the state from dataset with highest reward episode

    # Use a heuristic: take latent states from end of successful episodes
    # Episodes are 200 steps each, 5000 episodes
    # Pick states near the end of episodes (more likely close to target)
    ep_len = 200
    n_episodes = n_states // ep_len

    # Collect goal candidates: last 10 steps of each episode
    goal_indices = []
    for ep in range(n_episodes):
        for t in range(190, 200):
            goal_indices.append(ep * ep_len + t)
    goal_candidates = z_all[goal_indices]

    # Use the mean of these late-episode states as a soft goal
    # (many episodes won't reach the target, but some will)
    # Better: use the rewards to select
    rewards = torch.tensor(data.get('rewards', np.zeros(n_states)), dtype=torch.float32).to(DEVICE)

    # Find states with highest rewards (closest to target)
    if rewards.sum() > 0:
        top_idx = torch.topk(rewards, min(500, len(rewards))).indices
        z_goal = z_all[top_idx].mean(dim=0, keepdim=True)
        print(f"  Goal from top-500 reward states (reward sum: {rewards[top_idx].sum():.1f})")
    else:
        # Fallback: use late-episode states
        z_goal = goal_candidates.mean(dim=0, keepdim=True)
        print(f"  Goal from late-episode states (no rewards in dataset)")

    z_goal_norm = z_goal.norm()
    print(f"  Goal latent norm: {z_goal_norm:.2f}")

    # --- Create models ---
    Actor = make_actor()
    Critic = make_critic()

    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    critic_target = Critic().to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    n_actor_params = sum(p.numel() for p in actor.parameters())
    n_critic_params = sum(p.numel() for p in critic.parameters())
    print(f"\n{'='*60}")
    print(f"Phase 5: Dreamer Actor-Critic")
    print(f"Actor: {n_actor_params:,} params | Critic: {n_critic_params:,} params")
    print(f"Horizon: {HORIZON} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print(f"{'='*60}\n")

    # --- Training loop ---
    actor_losses_log, critic_losses_log = [], []
    reward_log, entropy_log = [], []
    best_avg_reward = -float('inf')

    for epoch in range(EPOCHS):
        # Sample real starting states
        idx = torch.randint(0, n_states, (BATCH_SIZE,), device=DEVICE)
        z_start = z_all[idx]

        # ── Imagine forward ──
        imagined_z = []       # [H+1, B, 1024]
        imagined_actions = [] # [H, B, 2]
        imagined_log_probs = []  # [H, B]
        imagined_rewards = [] # [H, B]

        z_t = z_start
        imagined_z.append(z_t)

        for h in range(HORIZON):
            action, log_prob = actor.sample(z_t)
            z_next = dynamics(z_t, action)

            # Reward: negative distance to goal
            dist_to_goal = torch.norm(z_next - z_goal.expand(BATCH_SIZE, -1), dim=-1)
            reward = -dist_to_goal / z_goal_norm  # normalize by goal norm

            imagined_z.append(z_next)
            imagined_actions.append(action)
            imagined_log_probs.append(log_prob)
            imagined_rewards.append(reward)

            z_t = z_next

        # Stack: shapes [H, B, ...]
        imagined_z = torch.stack(imagined_z)            # [H+1, B, 1024]
        imagined_rewards = torch.stack(imagined_rewards)  # [H, B]
        imagined_log_probs = torch.stack(imagined_log_probs)  # [H, B]

        # ── Compute TD-λ targets for critic ──
        with torch.no_grad():
            # Get value estimates for all imagined states
            v1_target, v2_target = critic_target(imagined_z[1:].reshape(-1, 1024))
            v_target = torch.min(v1_target, v2_target).reshape(HORIZON, BATCH_SIZE)

            # Bootstrap from final state
            v1_final, v2_final = critic_target(imagined_z[-1])
            v_bootstrap = torch.min(v1_final, v2_final)

            # TD-λ returns (computed backwards)
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
        # Re-imagine (need fresh graph through actor)
        z_t = z_start.detach()
        actor_reward_sum = torch.zeros(BATCH_SIZE, device=DEVICE)

        for h in range(HORIZON):
            action, log_prob_h = actor.sample(z_t)
            z_next = dynamics(z_t, action)

            # Value of next state (from frozen critic)
            with torch.no_grad():
                v1, v2 = critic(z_next)
                value = torch.min(v1, v2)

            # Reward for this step
            dist_to_goal = torch.norm(z_next - z_goal.expand(BATCH_SIZE, -1), dim=-1)
            reward = -dist_to_goal / z_goal_norm

            # Actor objective: maximize reward + γ*V(next) + entropy
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
            for p, p_target in zip(critic.parameters(), critic_target.parameters()):
                p_target.data.lerp_(p.data, TARGET_TAU)

        # ── Logging ──
        avg_reward = imagined_rewards.mean().item()
        avg_entropy = -imagined_log_probs.mean().item()

        actor_losses_log.append(actor_loss.item())
        critic_losses_log.append(critic_loss.item())
        reward_log.append(avg_reward)
        entropy_log.append(avg_entropy)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(actor.state_dict(), "/results/actor_dreamer.pt")
            torch.save(critic.state_dict(), "/results/critic_dreamer.pt")

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
                  f"Reward: {avg_reward:.4f} | "
                  f"Actor L: {actor_loss.item():.3f} | "
                  f"Critic L: {critic_loss.item():.4f} | "
                  f"Entropy: {avg_entropy:.3f} | "
                  f"Best: {best_avg_reward:.4f}")

    print(f"\nTraining done. Best imagined reward: {best_avg_reward:.4f}")

    # --- Save plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(reward_log)
    axes[0, 0].set_title("Avg Imagined Reward"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True)

    axes[0, 1].plot(entropy_log)
    axes[0, 1].set_title("Policy Entropy"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True)

    axes[1, 0].plot(actor_losses_log)
    axes[1, 0].set_title("Actor Loss"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)

    axes[1, 1].plot(critic_losses_log)
    axes[1, 1].set_title("Critic Loss"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True)

    plt.suptitle("Phase 5: Dreamer Actor-Critic Training")
    plt.tight_layout()
    plt.savefig("/results/dreamer_training.png", dpi=150)

    results_vol.commit()
    return {
        "best_imagined_reward": best_avg_reward,
        "final_reward": reward_log[-1],
        "actor_params": n_actor_params,
        "critic_params": n_critic_params,
    }


# ── Stage 2: Evaluate actor in real environment ────────────────────

@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def evaluate_actor():
    """Run the trained actor in dm_control and compare vs CEM baseline."""
    import torch
    import numpy as np
    import os, json
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
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
    print("Loading trained actor...")
    Actor = make_actor()
    actor = Actor().to(DEVICE)
    actor.load_state_dict(torch.load("/results/actor_dreamer.pt", weights_only=True))
    actor.eval()
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")

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

    # --- Eval environment ---
    env = suite.load("reacher", "easy", task_kwargs={'random': 42})
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs={
        'height': 224, 'width': 224, 'camera_id': 0
    })

    # --- Run actor ---
    print(f"\n{'='*60}")
    print("Phase 5: Actor Evaluation in Real Environment")
    print(f"{'='*60}")

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

        # Actor picks action (deterministic at eval time)
        with torch.no_grad():
            action = actor.act_deterministic(z_t).cpu().numpy().flatten()

        time_step = env.step(action)
        reward = time_step.reward or 0.0
        total_reward += reward
        rewards_log.append(reward)

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:3d} | LatDist: {lat_dist:.2f} | "
                  f"Reward: {reward:.1f} | Action: [{action[0]:.2f}, {action[1]:.2f}]")

    final_latent_dist = latent_distances[-1]
    min_latent_dist = min(latent_distances)
    improvement = (1 - final_latent_dist / init_latent_dist) * 100

    result = {
        "phase": "5",
        "model": "Dreamer Actor",
        "init_latent_dist": init_latent_dist,
        "final_latent_dist": final_latent_dist,
        "min_latent_dist": min_latent_dist,
        "improvement_pct": improvement,
        "total_env_reward": total_reward,
        "max_step_reward": max(rewards_log) if rewards_log else 0.0,
    }

    print(f"\n  Init: {init_latent_dist:.2f} → Final: {final_latent_dist:.2f} "
          f"(min: {min_latent_dist:.2f})")
    print(f"  Improvement: {improvement:.1f}% | Reward: {total_reward:.1f}")

    # Comparison
    print(f"\n{'='*60}")
    print("Phase 5 vs Baselines")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Improvement':>12} {'Reward':>8}")
    print("-" * 55)
    print(f"{'Phase 4e MLP CEM':<30} {'25.5%':>12} {'29.0':>8}")
    print(f"{'Phase 4e ResBlock CEM':<30} {'44.1%':>12} {'0.0':>8}")
    print(f"{'Phase 5 Dreamer Actor':<30} {improvement:>11.1f}% {total_reward:>8.1f}")

    # --- Save artifacts ---
    imageio.mimsave("/results/rollout_5_dreamer.mp4", frames, fps=20)

    plt.figure(figsize=(10, 5))
    plt.plot(latent_distances, label="Phase 5 Dreamer Actor", linewidth=2)
    plt.axhline(y=init_latent_dist, color='r', linestyle='--', alpha=0.5, label="Initial")
    plt.xlabel("Step"); plt.ylabel("Latent Distance to Goal")
    plt.title("Phase 5 — Dreamer Actor-Critic Evaluation")
    plt.legend(); plt.grid(True)
    plt.savefig("/results/latent_dist_5_dreamer.png", dpi=150)

    np.save("/results/latent_distances_5.npy", np.array(latent_distances))

    with open("/results/phase5_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Upload to HuggingFace
    print("\nUploading to HuggingFace...")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    upload_files = [
        ("/results/actor_dreamer.pt", "models/actor_dreamer.pt"),
        ("/results/dreamer_training.png", "results/dreamer_training.png"),
        ("/results/latent_dist_5_dreamer.png", "results/latent_dist_5_dreamer.png"),
        ("/results/phase5_results.json", "results/phase5_results.json"),
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
    print("Phase 5: Dreamer-Style Latent Actor-Critic")
    print("=" * 70)

    # Stage 1: Train
    print("\n[Stage 1] Training actor-critic in imagination...")
    train_result = train_dreamer.remote()
    print(f"  Training result: {train_result}")

    # Stage 2: Evaluate
    print("\n[Stage 2] Evaluating actor in real environment...")
    eval_result = evaluate_actor.remote()

    print("\n" + "=" * 70)
    print("Phase 5 Complete!")
    print("=" * 70)
    print(f"  Improvement: {eval_result['improvement_pct']:.1f}%")
    print(f"  Env Reward: {eval_result['total_env_reward']:.1f}")
    print(f"  (Baseline: Phase 4e MLP CEM = 29.0 reward)")
