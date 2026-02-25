"""
Phase 1a — Generate Expert Demos + Encode with V-JEPA 2
=========================================================
Runs on Modal A10G (~30-45 min, ~$0.75).

Pipeline per episode:
  1. Run scripted P-controller expert on dm_control reacher-easy
  2. Render 256×256 pixel frames at every step
  3. Build sliding 8-frame windows (V-JEPA input format)
  4. Batch-encode all windows with frozen V-JEPA 2
  5. Save (embeddings[1024], actions[2]) to Modal volume

Output: train_robots/data/reacher_easy_demos.npz
"""

import modal
from pathlib import Path
import os
import json
import numpy as np

app = modal.App("vjepa2-robot-demos")

model_cache = modal.Volume.from_name("vjepa2-model-cache", create_if_missing=True)
demo_vol    = modal.Volume.from_name("vjepa2-robot-demos", create_if_missing=True)

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
    ])
    .env({
        "MUJOCO_GL": "osmesa",
        "TRANSFORMERS_CACHE": "/cache/hf",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/cache": model_cache, "/demos": demo_vol},
)
def generate_and_encode(n_episodes: int = 500, max_steps: int = 200, batch_size: int = 32, epsilon: float = 0.3):
    """
    n_episodes: number of rollouts to generate
    max_steps:  environment steps per episode
    batch_size: V-JEPA encoding batch size
    epsilon:    probability of taking a totally random action (for exploration)
    """
    import torch
    from PIL import Image
    from torchvision import transforms
    from dm_control import suite
    from transformers import AutoModel

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load V-JEPA 2 (frozen) ─────────────────────────────────────────────
    print("[1] Loading V-JEPA 2...")
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()
    print(f"    {sum(p.numel() for p in vjepa.parameters()):,} params  |  frozen")

    ET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def encode_windows(frames: list, steps: int) -> np.ndarray:
        """
        Build sliding 8-frame windows for all timesteps in an episode [0..steps].
        Note: frames has length `steps + 1` (includes initial frame BEFORE step 0).
        Because we need z_{t+1}, we need to encode windows ending at t=0, 1, ..., steps.
        """
        embeddings = []
        # Precompute tensors for all frames 
        frame_tensors = [ET(Image.fromarray(f)) for f in frames]  # [steps+1, C, H, W]

        # We encode `steps + 1` total embeddings per episode (t=0 to t=steps)
        total_encodings = steps + 1
        
        for start_idx in range(0, total_encodings, batch_size):
            clips = []
            for t in range(start_idx, min(start_idx + batch_size, total_encodings)):
                # 8-frame window ending at index t
                window_start = max(0, t - 7)
                window = frame_tensors[window_start: t + 1]
                # Pad at front if fewer than 8 frames
                while len(window) < 8:
                    window = [window[0]] + window
                clips.append(torch.stack(window))  # [8, C, H, W]

            clips_t = torch.stack(clips).to(DEVICE, dtype=torch.float16)  # [B, 8, C, H, W]
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips_t)
                embs = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
            embeddings.append(embs)

        return np.concatenate(embeddings, axis=0)  # [steps + 1, 1024]

    def expert_policy(obs, env_spec, eps: float) -> np.ndarray:
        """
        Epsilon-greedy expert:
        - With prob (1-eps), use P-controller.
        - With prob eps, sample random uniform action.
        """
        if np.random.rand() < eps:
            return np.random.uniform(
                low=env_spec.minimum, 
                high=env_spec.maximum
            ).astype(np.float32)
        else:
            return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

    # ── 2. Generate demos ─────────────────────────────────────────────────────
    print(f"[2] Generating {n_episodes} episodes × {max_steps} steps (eps={epsilon})...")
    
    all_z_t = []
    all_a_t = []
    all_z_next = []
    all_rewards = []
    
    successes = 0

    for ep in range(n_episodes):
        env = suite.load("reacher", "easy", task_kwargs={"random": ep})
        action_spec = env.action_spec()
        
        time_step = env.reset()
        obs = time_step.observation

        # frames stores: state_0, state_1, ..., state_T (length T+1)
        frames  = [env.physics.render(height=256, width=256, camera_id=0).copy()]
        actions = []
        rewards = []

        for step in range(max_steps):
            action = expert_policy(obs, action_spec, epsilon)
            actions.append(action.copy())

            time_step = env.step(action)
            obs = time_step.observation
            rewards.append(float(time_step.reward or 0.0))

            frame = env.physics.render(height=256, width=256, camera_id=0)
            frames.append(frame.copy())

        # Batch-encode all windows for this episode (frames[0] through frames[T])
        ep_embs = encode_windows(frames, max_steps)  # [max_steps + 1, 1024]

        # Extract (z_t, a_t, z_next) triples
        z_t = ep_embs[:-1]       # [max_steps, 1024]
        z_next = ep_embs[1:]     # [max_steps, 1024]
        a_t = np.array(actions, dtype=np.float32) # [max_steps, 2]
        r_t = np.array(rewards, dtype=np.float32) # [max_steps]

        all_z_t.append(z_t)
        all_a_t.append(a_t)
        all_z_next.append(z_next)
        all_rewards.append(r_t)

        ep_reward = sum(rewards)
        if ep_reward > 150:
            successes += 1

        if ep % 50 == 0 or ep == n_episodes - 1:
            sr = successes / (ep + 1)
            print(f"    Episode {ep+1:4d}/{n_episodes}  |  eps={epsilon}  |  success rate: {sr:.1%}  |  ep_reward={ep_reward:.1f}")

    # ── 3. Save dataset ───────────────────────────────────────────────────────
    print("[3] Saving encoded dataset...")
    z_t    = np.concatenate(all_z_t, axis=0)     # [N, 1024]
    a_t    = np.concatenate(all_a_t, axis=0)     # [N, 2]
    z_next = np.concatenate(all_z_next, axis=0)  # [N, 1024]
    rewards = np.concatenate(all_rewards, axis=0) # [N]

    out_file = "/demos/reacher_easy_dynamics_demos.npz"
    np.savez_compressed(
        out_file,
        z_t=z_t,
        a_t=a_t,
        z_next=z_next,
        rewards=rewards,
    )

    meta = {
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "n_samples": int(len(z_t)),
        "expert_success_rate": float(successes / n_episodes),
        "epsilon": epsilon,
        "embedding_dim": 1024,
        "action_dim": 2,
        "dataset_mb": float(z_t.nbytes * 2 / 1e6), # z_t + z_next
    }
    
    with open("/demos/dynamics_demo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    demo_vol.commit()
    print(f"    Saved {len(z_t):,} transition triples  |  success rate: {meta['expert_success_rate']:.1%}")
    return meta


@app.local_entrypoint()
def main():
    import subprocess

    print("=" * 70)
    print("Phase 2: Generating diverse exploration demos + encoding with V-JEPA 2")
    print("=" * 70)

    # 500 eps is 100k transitions. Epsilon 0.3 means 30% random actions to explore state space.
    meta = generate_and_encode.remote(n_episodes=500, max_steps=200, epsilon=0.3)

    print("\n=== Demo generation complete ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Download dataset locally
    out_dir = Path("train_robots/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["reacher_easy_dynamics_demos.npz", "dynamics_demo_meta.json"]:
        p = str(out_dir / fname)
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-robot-demos", fname, p],
                check=True
            )
            print(f"  Downloaded → {p}")
        except Exception:
            print(f"  ⚠ Run manually: modal volume get vjepa2-robot-demos {fname} {p}")

