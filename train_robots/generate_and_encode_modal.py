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
def generate_and_encode(n_episodes: int = 500, max_steps: int = 200, batch_size: int = 32):
    """
    n_episodes: number of expert rollouts to generate
    max_steps:  environment steps per episode
    batch_size: V-JEPA encoding batch size (per-episode windowed clips)
    """
    import os, json
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from dm_control import suite
    from transformers import AutoModel

    DEVICE = "cuda"

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
        Build sliding 8-frame windows for all timesteps in an episode,
        batch-encode with V-JEPA, return ndarray [steps, 1024].
        """
        embeddings = []
        # Precompute tensors for all frames once
        frame_tensors = [ET(Image.fromarray(f)) for f in frames]  # each [C, H, W]

        for start_idx in range(0, steps, batch_size):
            clips = []
            for t in range(start_idx, min(start_idx + batch_size, steps)):
                # 8-frame window ending at t, padded at start of episode
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

        return np.concatenate(embeddings, axis=0)  # [steps, 1024]

    def scripted_expert(obs) -> np.ndarray:
        """
        Simple proportional controller in end-effector space.
        obs['to_target']: 2D Cartesian delta from fingertip to target.
        Action space: 2 joint torques in [-1, 1].
        """
        return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

    # ── 2. Generate demos ─────────────────────────────────────────────────────
    print(f"[2] Generating {n_episodes} episodes × {max_steps} steps...")
    all_embeddings = []
    all_actions    = []
    all_rewards    = []
    successes      = 0

    for ep in range(n_episodes):
        env = suite.load("reacher", "easy", task_kwargs={"random": ep})
        time_step = env.reset()
        obs = time_step.observation

        # Collect raw frames + actions for this episode
        frames  = []
        actions = []
        rewards = []

        first_frame = env.physics.render(height=256, width=256, camera_id=0)
        frames.append(first_frame.copy())

        for step in range(max_steps):
            action = scripted_expert(obs)
            actions.append(action.copy())

            time_step = env.step(action)
            obs = time_step.observation
            rewards.append(float(time_step.reward or 0.0))

            frame = env.physics.render(height=256, width=256, camera_id=0)
            frames.append(frame.copy())

        # Batch-encode all windows for this episode (frames[0..max_steps])
        ep_embs = encode_windows(frames, max_steps)  # [max_steps, 1024]

        all_embeddings.append(ep_embs)
        all_actions.append(np.array(actions, dtype=np.float32))
        all_rewards.append(np.array(rewards, dtype=np.float32))

        ep_reward = sum(rewards)
        if ep_reward > 150:
            successes += 1

        if ep % 50 == 0 or ep == n_episodes - 1:
            sr = successes / (ep + 1)
            print(f"    Episode {ep+1:4d}/{n_episodes}  |  expert success rate: {sr:.1%}  |  ep_reward={ep_reward:.1f}")

    # ── 3. Save dataset ───────────────────────────────────────────────────────
    print("[3] Saving encoded dataset...")
    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 1024]
    actions    = np.concatenate(all_actions, axis=0)     # [N, 2]
    rewards    = np.concatenate(all_rewards, axis=0)     # [N]

    np.savez_compressed(
        "/demos/reacher_easy_demos.npz",
        embeddings=embeddings,
        actions=actions,
        rewards=rewards,
    )

    meta = {
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "n_samples": int(len(embeddings)),
        "expert_success_rate": float(successes / n_episodes),
        "embedding_dim": 1024,
        "action_dim": 2,
        "dataset_mb": float(embeddings.nbytes / 1e6),
    }
    with open("/demos/demo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    demo_vol.commit()
    print(f"    Saved {len(embeddings):,} samples  |  {meta['dataset_mb']:.0f} MB  |  success rate: {meta['expert_success_rate']:.1%}")
    return meta


@app.local_entrypoint()
def main():
    import subprocess
    from pathlib import Path

    print("=" * 60)
    print("Phase 1a: Generating expert demos + encoding with V-JEPA 2")
    print("=" * 60)

    meta = generate_and_encode.remote(n_episodes=500, max_steps=200)

    print("\n=== Demo generation complete ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Download dataset locally
    out = Path("train_robots/data")
    out.mkdir(parents=True, exist_ok=True)

    for fname in ["reacher_easy_demos.npz", "demo_meta.json"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-robot-demos", fname, str(out / fname)],
                check=True
            )
            print(f"  Downloaded → train_robots/data/{fname}")
        except Exception:
            print(f"  ⚠ Run manually: modal volume get vjepa2-robot-demos {fname} train_robots/data/{fname}")
