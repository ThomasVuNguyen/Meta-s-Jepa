"""
Phase 1c — Evaluate BC Policy in dm_control
=============================================
Runs on Modal A10G (~20-30 min, ~$0.40).

Rolls out the trained BCPolicy for n_episodes in dm_control reacher-easy,
encoding observations with frozen V-JEPA 2 at every step.

Compares:
  - V-JEPA BC policy  (our approach)
  - Scripted expert   (upper bound)
  - Random policy     (lower bound)

Outputs:
  train_robots/results/eval_results.json
  train_robots/results/eval_results.png
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-robot-eval")

model_cache = modal.Volume.from_name("vjepa2-model-cache", create_if_missing=True)
demo_vol    = modal.Volume.from_name("vjepa2-robot-demos",  create_if_missing=True)

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
        "matplotlib",
    ])
    .env({
        "MUJOCO_GL": "osmesa",
        "TRANSFORMERS_CACHE": "/cache/hf",
    })
)


# ── BC Policy definition (must match train_bc.py exactly) ─────────────────────
import torch.nn as _nn

class BCPolicy(_nn.Module):
    def __init__(self, obs_dim=1024, action_dim=2, hidden=256):
        super().__init__()
        self.net = _nn.Sequential(
            _nn.Linear(obs_dim, hidden),
            _nn.LayerNorm(hidden),
            _nn.ReLU(),
            _nn.Linear(hidden, 64),
            _nn.ReLU(),
            _nn.Linear(64, action_dim),
            _nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

def make_policy(obs_dim=1024, action_dim=2, hidden=256):
    return BCPolicy(obs_dim, action_dim, hidden)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": model_cache, "/demos": demo_vol},
)
def evaluate(n_episodes: int = 100, max_steps: int = 200):
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    from collections import deque
    from PIL import Image
    from torchvision import transforms
    from dm_control import suite
    from transformers import AutoModel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"

    # ── Load V-JEPA 2 ─────────────────────────────────────────────────────────
    print("[1] Loading V-JEPA 2...")
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

    def encode_buffer(frame_buf: deque) -> np.ndarray:
        """Encode an 8-frame buffer → 1024-dim embedding (single call)."""
        tensors = [ET(Image.fromarray(f)) for f in list(frame_buf)]
        clip = torch.stack(tensors).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=clip)
            emb = out.last_hidden_state.mean(dim=1).cpu().float()
        return emb  # [1, 1024]

    # ── Load BC policy ────────────────────────────────────────────────────────
    print("[2] Loading BC policy...")
    ckpt   = torch.load("/demos/bc_policy.pt", map_location="cpu")
    policy = make_policy()
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    X_mean = ckpt["X_mean"]
    X_std  = ckpt["X_std"]

    def vjepa_bc_action(frame_buf: deque) -> np.ndarray:
        emb  = encode_buffer(frame_buf)
        norm = (emb - X_mean) / X_std
        with torch.no_grad():
            act = policy(norm).numpy()[0]
        return act

    def expert_action(obs) -> np.ndarray:
        return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

    def random_action() -> np.ndarray:
        return np.random.uniform(-1, 1, size=2).astype(np.float32)

    SUCCESS_REWARD_THRESHOLD = 150  # cumulative reward for a "successful" episode

    # ── Run rollouts ──────────────────────────────────────────────────────────
    conditions = ["vjepa_bc", "expert", "random"]
    results = {}

    for cond in conditions:
        print(f"\n[3] Evaluating: {cond} ...")
        successes = 0
        ep_rewards = []

        for ep in range(n_episodes):
            env = suite.load("reacher", "easy", task_kwargs={"random": ep + 5000})
            obs = env.reset().observation

            frame_buf = deque(maxlen=8)
            first_frame = env.physics.render(height=256, width=256, camera_id=0)
            for _ in range(8):
                frame_buf.append(first_frame.copy())

            ep_reward = 0.0
            for _ in range(max_steps):
                if cond == "vjepa_bc":
                    action = vjepa_bc_action(frame_buf)
                elif cond == "expert":
                    action = expert_action(obs)
                else:
                    action = random_action()

                ts  = env.step(action)
                obs = ts.observation
                ep_reward += ts.reward or 0.0

                frame = env.physics.render(height=256, width=256, camera_id=0)
                frame_buf.append(frame)

            ep_rewards.append(ep_reward)
            if ep_reward > SUCCESS_REWARD_THRESHOLD:
                successes += 1

            if ep % 20 == 0:
                print(f"    {ep}/{n_episodes}  success={successes/(ep+1):.1%}  last_reward={ep_reward:.1f}")

        results[cond] = {
            "success_rate":  round(successes / n_episodes, 4),
            "mean_reward":   round(float(np.mean(ep_rewards)), 2),
            "n_episodes":    n_episodes,
        }
        print(f"  ✓ {cond}: success={results[cond]['success_rate']:.1%}  mean_reward={results[cond]['mean_reward']:.1f}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    with open("/demos/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels  = ["V-JEPA BC\n(ours)", "Scripted\nExpert", "Random\nPolicy"]
    values  = [results[c]["success_rate"] for c in conditions]
    colors  = ["#4fc3f7", "#a5d6a7", "#ef9a9a"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", color="white", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate", color="white", fontsize=12)
    ax.set_title(
        "dm_control reacher-easy — Policy Evaluation\n"
        f"(n={n_episodes} episodes, success = cumulative reward > {SUCCESS_REWARD_THRESHOLD})",
        color="white", fontsize=12
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    plt.tight_layout()
    plt.savefig("/demos/eval_results.png", dpi=120, facecolor="#111")

    demo_vol.commit()
    return results


@app.local_entrypoint()
def main():
    import subprocess
    from pathlib import Path

    # Upload trained policy to Modal volume
    policy_path = Path("train_robots/data/bc_policy.pt")
    if not policy_path.exists():
        raise FileNotFoundError(
            "bc_policy.pt not found. Run Phase 1b first: python3 train_robots/train_bc.py"
        )

    print("Uploading BC policy to Modal volume...")
    subprocess.run(
        ["modal", "volume", "put", "vjepa2-robot-demos",
         str(policy_path), "bc_policy.pt"],
        check=False  # OK if already exists
    )

    print("Running evaluation rollouts...")
    results = evaluate.remote(n_episodes=100)

    print("\n=== Evaluation Results ===")
    for cond, metrics in results.items():
        sr = metrics["success_rate"]
        mr = metrics["mean_reward"]
        print(f"  {cond:12s}: success={sr:.1%}  mean_reward={mr:.1f}")

    # Download results locally
    out = Path("train_robots/results")
    out.mkdir(parents=True, exist_ok=True)
    for fname in ["eval_results.json", "eval_results.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-robot-demos", fname, str(out / fname)],
                check=True
            )
            print(f"  Downloaded → train_robots/results/{fname}")
        except Exception:
            print(f"  ⚠ Run manually: modal volume get vjepa2-robot-demos {fname} train_robots/results/{fname}")
