"""
Phase 3 — Train Action-Conditioned Dynamics Predictor
=====================================================
This script trains the core "World Model" component: a neural network
that learns to predict the next latent state given the current latent state
and an action: f(z_t, a_t) -> z_{t+1}.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DynamicsPredictor(nn.Module):
    """
    A simple MLP that predicts the change in latent state (residual prediction).
    Input: concat(z_t [1024], a_t [2]) -> 1026
    Output: delta_z [1024]
    """
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
        # We predict the residual: z_next = z_t + f(z_t, a_t)
        # This makes it easier for the network to learn identity when actions do nothing
        x = torch.cat([z_t, a_t], dim=-1)
        delta_z = self.net(x)
        return z_t + delta_z

def load_data(data_path: str):
    print(f"Loading dataset from {data_path}...")
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset {data_path} not found.")
        print("Please ensure `generate_and_encode_modal.py` has finished running.")
        exit(1)
        
    z_t    = torch.tensor(data['z_t'], dtype=torch.float32)
    a_t    = torch.tensor(data['a_t'], dtype=torch.float32)
    z_next = torch.tensor(data['z_next'], dtype=torch.float32)
    
    # 80/20 train/val split
    n = len(z_t)
    train_size = int(0.8 * n)
    
    train_ds = TensorDataset(z_t[:train_size], a_t[:train_size], z_next[:train_size])
    val_ds   = TensorDataset(z_t[train_size:], a_t[train_size:], z_next[train_size:])
    
    return train_ds, val_ds

def train():
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = "train_robots/data/reacher_easy_dynamics_demos.npz"
    train_ds, val_ds = load_data(data_path)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DynamicsPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss is often better for latents than pure MSE
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"\nTraining Dynamics Predictor on {DEVICE}")
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        epoch_train_loss = 0.0
        for z_t, a_t, z_next in train_loader:
            z_t, a_t, z_next = z_t.to(DEVICE), a_t.to(DEVICE), z_next.to(DEVICE)
            
            optimizer.zero_grad()
            z_next_pred = model(z_t, a_t)
            
            loss = criterion(z_next_pred, z_next)
            loss.backward()
            
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item() * z_t.size(0)
            
        epoch_train_loss /= len(train_ds)
        train_losses.append(epoch_train_loss)
        
        # -- Validate --
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for z_t, a_t, z_next in val_loader:
                z_t, a_t, z_next = z_t.to(DEVICE), a_t.to(DEVICE), z_next.to(DEVICE)
                z_next_pred = model(z_t, a_t)
                loss = criterion(z_next_pred, z_next)
                epoch_val_loss += loss.item() * z_t.size(0)
                
        epoch_val_loss /= len(val_ds)
        val_losses.append(epoch_val_loss)
        scheduler.step()
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            Path("train_robots/models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "train_robots/models/dynamics_predictor.pt")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.5f}")
    print("Saved best model to train_robots/models/dynamics_predictor.pt")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.title("Dynamics Predictor Training")
    plt.legend()
    plt.grid(True)
    Path("train_robots/results").mkdir(parents=True, exist_ok=True)
    plt.savefig("train_robots/results/dynamics_loss_curve.png")
    print("Saved loss curve to train_robots/results/dynamics_loss_curve.png")


if __name__ == "__main__":
    train()
