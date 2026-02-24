"""
Phase 1b — Train Behavior Cloning Policy
==========================================
Runs locally on CPU (~2 min). Requires train_robots/data/reacher_easy_demos.npz.

Model: 3-layer MLP (1024 → 256 → 64 → 2) with LayerNorm + Tanh output.
Loss:  MSE on expert joint velocities.

Outputs:
  train_robots/data/bc_policy.pt     — model weights + input normalizer
  train_robots/data/bc_loss.png      — training loss curve
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("train_robots/data/reacher_easy_demos.npz")
OUT_DIR   = Path("train_robots/data")


class BCPolicy(nn.Module):
    """
    Behavior cloning policy: V-JEPA embedding → joint velocities.
    Small enough to train on CPU in ~2 minutes.
    """
    def __init__(self, obs_dim: int = 1024, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),   # actions are in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(epochs: int = 100, batch_size: int = 512, lr: float = 3e-4):
    # ── Load dataset ───────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Run Phase 1a first: modal run train_robots/generate_and_encode_modal.py"
        )

    print(f"Loading dataset from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    X = torch.tensor(data["embeddings"], dtype=torch.float32)
    y = torch.tensor(data["actions"],    dtype=torch.float32)
    print(f"  Samples: {X.shape[0]:,}  |  X={tuple(X.shape)}  |  y={tuple(y.shape)}")

    # ── Normalise inputs ───────────────────────────────────────────────────────
    X_mean = X.mean(dim=0, keepdim=True)
    X_std  = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std

    # ── Train / val split ──────────────────────────────────────────────────────
    n     = len(X_norm)
    split = int(0.8 * n)
    perm  = torch.randperm(n)
    tr, va = perm[:split], perm[split:]

    train_dl = DataLoader(TensorDataset(X_norm[tr], y[tr]), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_norm[va], y[va]), batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = BCPolicy()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs...")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                v_loss += loss_fn(model(xb), yb).item()

        t_loss /= len(train_dl)
        v_loss /= len(val_dl)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        sched.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  train={t_loss:.4f}  |  val={v_loss:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std":  X_std,
        "obs_dim": 1024,
        "action_dim": 2,
        "hidden": 256,
    }, OUT_DIR / "bc_policy.pt")
    print(f"\nSaved model → {OUT_DIR}/bc_policy.pt")

    # ── Loss plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")
    ax.plot(train_losses, color="#4fc3f7", label="Train MSE")
    ax.plot(val_losses,   color="#ef9a9a", label="Val MSE",  linestyle="--")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.set_title("BC Policy — V-JEPA embeddings → joint velocities", color="white")
    ax.legend(facecolor="#222", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bc_loss.png", dpi=120, facecolor="#111")
    print(f"Saved loss plot → {OUT_DIR}/bc_loss.png")

    return model, X_mean, X_std


if __name__ == "__main__":
    train()
