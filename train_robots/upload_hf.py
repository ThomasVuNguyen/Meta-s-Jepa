"""Upload datasets and models to HuggingFace Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "ThomasTheMaker/vjepa2-robot-multitask"
DATA = Path("train_robots/data")
MODELS = Path("train_robots/models/models")

api = HfApi(token=os.environ.get("HF_TOKEN"))

# Create repo (or use existing)
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=os.environ.get("HF_TOKEN"))
    print(f"Repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo exists or error: {e}")

# Upload datasets
for f in sorted(DATA.glob("*_1k.npz")):
    print(f"Uploading {f.name} ({f.stat().st_size/1e9:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=f"data/{f.name}",
        repo_id=REPO_ID,
    )
    print(f"  ✓ {f.name}")

# Upload models
for task_dir in sorted(MODELS.iterdir()):
    if not task_dir.is_dir():
        continue
    for pt in sorted(task_dir.glob("*.pt")):
        print(f"Uploading {task_dir.name}/{pt.name}...")
        api.upload_file(
            path_or_fileobj=str(pt),
            path_in_repo=f"models/{task_dir.name}/{pt.name}",
            repo_id=REPO_ID,
        )
        print(f"  ✓ {task_dir.name}/{pt.name}")

# Upload README
readme = f"""---
license: mit
tags:
  - robotics
  - vjepa2
  - dm_control
  - world-model
  - teach-by-showing
---

# V-JEPA 2 Robot Multi-Task Dataset & Models

Vision-based robot control data using **V-JEPA 2** (ViT-L) latent representations
from DeepMind Control Suite environments.

## 📊 Dataset

| Task | Episodes | Transitions | Latent Dim | Action Dim | Success Rate |
|------|----------|-------------|------------|------------|-------------|
| reacher_easy | 1,000 | 200,000 | 1024 | 2 | 28.9% |
| point_mass_easy | 1,000 | 200,000 | 1024 | 2 | 0.6% |
| cartpole_swingup | 1,000 | 200,000 | 1024 | 1 | 0.0% |

Each `.npz` file contains:
- `z_t` — V-JEPA 2 latent state embeddings (N × 1024)
- `a_t` — actions taken (N × action_dim)
- `z_next` — next-state latent embeddings (N × 1024)
- `rewards` — per-step rewards (N,)

## 🤖 Models

For each task, we provide:
- **5× Dynamics Ensemble** — `dyn_0.pt` to `dyn_4.pt` (MLP: z + a → z_next, ~1.58M params each)
- **1× Reward Model** — `reward.pt` (MLP: z + a → reward, ~329K params)

### Architecture
- Dynamics: `Linear(1024+a_dim, 512) → LN → ReLU → ×3 → Linear(512, 1024)` + residual connection
- Reward: `Linear(1024+a_dim, 256) → ReLU → ×2 → Linear(256, 1)`
- Ensemble diversity (weight cosine sim): ~0.60

## 🏗️ How It Was Built

1. Expert policies collect episodes in dm_control environments
2. Each frame rendered at 224×224, encoded with V-JEPA 2 ViT-L (8-frame sliding windows)
3. Dynamics ensemble trained with random data splits + different seeds
4. Reward model trained to predict per-step rewards from z_t + a_t

## 📈 Training Details

- **GPU:** NVIDIA A100-SXM4-80GB (Prime Intellect)
- **Total time:** 5.4 hours
- **Total cost:** ~$7
- **Dynamics val loss:** ~0.0008 (reacher, point_mass), ~0.0002 (cartpole)
- **Temporal coherence:** >0.998 for all tasks

## 🎯 Purpose

These world models are designed for **"teach-by-showing"** — demonstrating a task via video,
then using the learned dynamics + CEM planning to reproduce the shown behavior.
"""

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
)
print(f"\n✅ Done! https://huggingface.co/{REPO_ID}")
