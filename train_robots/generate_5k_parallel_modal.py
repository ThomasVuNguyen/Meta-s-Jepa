"""
Phase 2b — Large-Scale Data Collection (5000 episodes, parallelized)
====================================================================
Runs on Modal: 10 × A10G workers in parallel (~1 hour wall time, ~$7-8).

Each worker:
  1. Generates 500 episodes with epsilon-greedy P-controller
  2. Encodes all frames through frozen V-JEPA 2
  3. Saves (z_t, a_t, z_next) triples to Modal volume

Local entrypoint merges all shards into a single dataset.

Output: train_robots/data/reacher_easy_5k.npz (~5GB, ~1M transitions)
"""

import modal
from pathlib import Path
import json
import numpy as np

app = modal.App("vjepa2-data-5k")

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
def generate_shard(shard_id: int, episodes_per_shard: int = 500,
                   max_steps: int = 200, batch_size: int = 32,
                   epsilon: float = 0.3):
    """
    Generate and encode one shard of episodes.
    Each shard gets unique episode seeds: [shard_id*eps_per_shard, (shard_id+1)*eps_per_shard)
    """
    import torch
    from PIL import Image
    from torchvision import transforms
    from dm_control import suite
    from transformers import AutoModel

    DEVICE = "cuda"
    seed_offset = shard_id * episodes_per_shard

    # Load V-JEPA 2
    print(f"[Shard {shard_id}] Loading V-JEPA 2...")
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()

    ET = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def encode_windows(frames, steps):
        embeddings = []
        frame_tensors = [ET(Image.fromarray(f)) for f in frames]
        total_encodings = steps + 1

        for start_idx in range(0, total_encodings, batch_size):
            clips = []
            for t in range(start_idx, min(start_idx + batch_size, total_encodings)):
                window_start = max(0, t - 7)
                window = frame_tensors[window_start: t + 1]
                while len(window) < 8:
                    window = [window[0]] + window
                clips.append(torch.stack(window))

            clips_t = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips_t)
                embs = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
            embeddings.append(embs)

        return np.concatenate(embeddings, axis=0)

    def expert_policy(obs, action_spec, eps):
        if np.random.rand() < eps:
            return np.random.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum
            ).astype(np.float32)
        else:
            return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

    # Generate episodes
    print(f"[Shard {shard_id}] Generating {episodes_per_shard} episodes "
          f"(seeds {seed_offset}-{seed_offset + episodes_per_shard - 1})...")

    all_z_t = []
    all_a_t = []
    all_z_next = []
    all_rewards = []
    successes = 0

    for ep_local in range(episodes_per_shard):
        ep_global = seed_offset + ep_local
        env = suite.load("reacher", "easy", task_kwargs={"random": ep_global})
        action_spec = env.action_spec()

        time_step = env.reset()
        obs = time_step.observation

        frames = [env.physics.render(height=224, width=224, camera_id=0).copy()]
        actions = []
        rewards = []

        for step in range(max_steps):
            action = expert_policy(obs, action_spec, epsilon)
            actions.append(action.copy())
            time_step = env.step(action)
            obs = time_step.observation
            rewards.append(float(time_step.reward or 0.0))
            frames.append(env.physics.render(height=224, width=224, camera_id=0).copy())

        # Encode
        ep_embs = encode_windows(frames, max_steps)
        z_t = ep_embs[:-1]
        z_next = ep_embs[1:]
        a_t = np.array(actions, dtype=np.float32)
        r_t = np.array(rewards, dtype=np.float32)

        all_z_t.append(z_t)
        all_a_t.append(a_t)
        all_z_next.append(z_next)
        all_rewards.append(r_t)

        ep_reward = sum(rewards)
        if ep_reward > 150:
            successes += 1

        if (ep_local + 1) % 50 == 0 or ep_local == 0:
            sr = successes / (ep_local + 1)
            print(f"  [Shard {shard_id}] Episode {ep_local+1:4d}/{episodes_per_shard} | "
                  f"SR: {sr:.1%} | reward={ep_reward:.1f}")

    # Save shard
    z_t    = np.concatenate(all_z_t, axis=0)
    a_t    = np.concatenate(all_a_t, axis=0)
    z_next = np.concatenate(all_z_next, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    shard_file = f"/demos/shard_{shard_id:03d}.npz"
    np.savez_compressed(shard_file, z_t=z_t, a_t=a_t, z_next=z_next, rewards=rewards)

    meta = {
        "shard_id": shard_id,
        "n_episodes": episodes_per_shard,
        "n_transitions": int(len(z_t)),
        "success_rate": float(successes / episodes_per_shard),
        "seed_range": [seed_offset, seed_offset + episodes_per_shard],
    }
    print(f"[Shard {shard_id}] Done. {len(z_t):,} transitions | SR: {meta['success_rate']:.1%}")

    demo_vol.commit()
    return meta


@app.local_entrypoint()
def main():
    import subprocess

    N_SHARDS = 10
    EPISODES_PER_SHARD = 500
    TOTAL = N_SHARDS * EPISODES_PER_SHARD

    print("=" * 70)
    print(f"Phase 2b: Generating {TOTAL:,} episodes across {N_SHARDS} parallel workers")
    print("=" * 70)

    # Launch all shards in parallel using .map()
    shard_ids = list(range(N_SHARDS))
    results = list(generate_shard.map(shard_ids))

    print("\n" + "=" * 70)
    print("All shards complete!")
    for r in results:
        print(f"  Shard {r['shard_id']}: {r['n_transitions']:,} transitions, "
              f"SR={r['success_rate']:.1%}")

    total_transitions = sum(r['n_transitions'] for r in results)
    avg_sr = np.mean([r['success_rate'] for r in results])
    print(f"\nTotal: {total_transitions:,} transitions | Avg SR: {avg_sr:.1%}")

    # Download shards and merge locally
    out_dir = Path("train_robots/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading and merging shards...")
    all_z_t, all_a_t, all_z_next, all_rewards = [], [], [], []

    for i in range(N_SHARDS):
        shard_file = f"shard_{i:03d}.npz"
        local_path = str(out_dir / shard_file)
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-robot-demos", shard_file, local_path],
                check=True
            )
            data = np.load(local_path)
            all_z_t.append(data['z_t'])
            all_a_t.append(data['a_t'])
            all_z_next.append(data['z_next'])
            all_rewards.append(data['rewards'])
            print(f"  Shard {i}: {len(data['z_t']):,} transitions")
        except Exception as e:
            print(f"  ⚠ Shard {i} download failed: {e}")

    # Merge
    z_t    = np.concatenate(all_z_t, axis=0)
    a_t    = np.concatenate(all_a_t, axis=0)
    z_next = np.concatenate(all_z_next, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    merged_path = str(out_dir / "reacher_easy_5k.npz")
    np.savez_compressed(merged_path, z_t=z_t, a_t=a_t, z_next=z_next, rewards=rewards)

    print(f"\nMerged dataset: {merged_path}")
    print(f"  Transitions: {len(z_t):,}")
    print(f"  z_t shape: {z_t.shape}")
    print(f"  File size: {Path(merged_path).stat().st_size / 1e9:.2f} GB")

    # Save meta
    meta = {
        "n_episodes": TOTAL,
        "n_shards": N_SHARDS,
        "n_transitions": int(len(z_t)),
        "avg_success_rate": float(avg_sr),
        "shard_results": results,
    }
    with open(str(out_dir / "reacher_easy_5k_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup shard files
    for i in range(N_SHARDS):
        shard_path = out_dir / f"shard_{i:03d}.npz"
        if shard_path.exists():
            shard_path.unlink()
    print("Cleaned up shard files.")

    print(f"\n{'=' * 70}")
    print(f"Phase 2b complete: {len(z_t):,} transitions from {TOTAL} episodes")
    print(f"{'=' * 70}")
