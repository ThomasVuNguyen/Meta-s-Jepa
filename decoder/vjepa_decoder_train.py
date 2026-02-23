"""
vjepa_decoder_train.py
======================
Trains a CNN decoder that maps V-JEPA 2 latent patch tokens
→ reconstructed video frames.

Architecture:
  Encoder (frozen):  facebook/vjepa2-vitl-fpc64-256
                     Input:  [B, C, 8, 256, 256]
                     Output: [B, 1024, 1024]  (1024 spatiotemporal tokens)

  Decoder (trained): Token embedding grid → upsampled RGB frame
                     Input:  [B, 1024, 1024]
                     Output: [B, 3, 256, 256]

Loss:   MSE + gradient sharpness term
Run On: GPU with ≥8GB VRAM (e.g. Colab T4)

Usage:
  pip install transformers accelerate yt-dlp opencv-python-headless
  python vjepa_decoder_train.py
"""

import os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ── 0. Config ──────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = Path("/tmp/vjepa_decoder_data")
MODEL_SAVE  = "/tmp/vjepa_decoder_v1.pt"
EPOCHS      = 30
BATCH_SIZE  = 4
LR          = 2e-4
HIDDEN_DIM  = 512
N_CLUSTERS  = 6

print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    print("⚠  No GPU found — training will be very slow. Use Colab T4.")


# ── 1. Load V-JEPA 2 Encoder (frozen) ─────────────────────────────────────
print("\n[1] Loading V-JEPA 2 encoder...")
from transformers import AutoModel

encoder = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    trust_remote_code=True
)
encoder = encoder.to(DEVICE, dtype=torch.float16).eval()
for p in encoder.parameters():
    p.requires_grad = False
print("    Encoder loaded and frozen ✓")


# ── 2. Data Pipeline ────────────────────────────────────────────────────────
enc_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tgt_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),   # [0, 1] range, no normalisation → target pixel values
])

# Videos to download for training data
TRAINING_VIDEOS = [
    ("https://www.youtube.com/watch?v=_FjuOVeahA8", "bbb",    30),  # Big Buck Bunny
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ", "nature", 30),  # Nature footage
    ("https://www.youtube.com/watch?v=3JZ_D3ELwOQ", "ocean",  30),  # Ocean
    ("https://www.youtube.com/watch?v=YE7VzlLtp-4", "city",   20),  # City timelapse
]

def download_video(url, name, duration_s):
    """Download a short video clip using yt-dlp."""
    import cv2
    path = f"/tmp/train_{name}.mp4"
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"    {name}: already downloaded")
        return path
    print(f"    Downloading {name} ({duration_s}s)...")
    result = subprocess.run([
        "yt-dlp", "--quiet",
        "-f", "bestvideo[height<=360][ext=mp4]/best[height<=360]",
        "--download-sections", f"*0:00-0:{duration_s:02d}",
        "-o", path, url
    ], capture_output=True, timeout=90)
    if os.path.exists(path):
        print(f"    {name}: {os.path.getsize(path)//1024}KB ✓")
        return path
    print(f"    {name}: download failed, skipping")
    return None


def encode_video_to_pairs(video_path, clip_id_offset, n_clips=15):
    """
    Extract 8-frame clips from a video at multiple positions.
    For each clip: encode with V-JEPA 2, save (embedding, middle_frame) pair.
    Returned: number of clips encoded.
    """
    import cv2
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0

    starts = np.linspace(0, max(0, total_frames - 50), n_clips, dtype=int)

    for ci, start in enumerate(starts):
        frames_enc = []   # encoder input (normalised)
        frames_tgt = []   # target pixel values

        for t in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start + t * 4))
            ret, fr = cap.read()
            if not ret:
                break
            pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            frames_enc.append(enc_transform(pil))
            frames_tgt.append(tgt_transform(pil))

        if len(frames_enc) < 8:
            continue

        # Build video tensor [1, C, 8, H, W]
        vid = torch.stack(frames_enc).permute(1, 0, 2, 3).unsqueeze(0)
        vid = vid.to(DEVICE, dtype=torch.float16)

        with torch.no_grad():
            emb = encoder(pixel_values_videos=vid).last_hidden_state  # [1, 1024, 1024]
            emb = emb[0].cpu().float()  # [1024, 1024]

        target = frames_tgt[3]  # middle frame as reconstruction target [3, H, W]

        path = DATASET_DIR / f"pair_{clip_id_offset + ci:05d}.pt"
        torch.save({"embedding": emb, "target": target}, path)
        saved += 1

    cap.release()
    return saved


print("\n[2] Building training dataset...")
total_pairs = 0
clip_offset = 0
for url, name, dur in TRAINING_VIDEOS:
    vpath = download_video(url, name, dur)
    if vpath:
        n = encode_video_to_pairs(vpath, clip_offset, n_clips=18)
        print(f"      {name}: {n} pairs encoded")
        total_pairs += n
        clip_offset += 20

print(f"    Total: {total_pairs} training pairs")


# ── 3. Dataset & Dataloader ─────────────────────────────────────────────────
class EmbeddingFrameDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("pair_*.pt"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        return d["embedding"], d["target"]   # [1024, 1024], [3, 256, 256]

dataset = EmbeddingFrameDataset(DATASET_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"\n[3] Dataset: {len(dataset)} samples  |  {len(loader)} batches/epoch")


# ── 4. CNN Decoder Model ────────────────────────────────────────────────────
class VJEPADecoder(nn.Module):
    """
    Maps V-JEPA 2 patch tokens [B, 1024, 1024] → RGB frame [B, 3, 256, 256].

    Token layout: 1024 tokens = T=4 temporal positions × 16×16 spatial grid.
    We mean-pool over T, giving a 16×16 spatial map, then progressively upsample.
    """
    def __init__(self, token_dim=1024, hidden=512):
        super().__init__()
        # Per-token projection
        self.proj = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )

        def up(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1),
                nn.GroupNorm(8, oc),
                nn.GELU(),
                nn.Conv2d(oc, oc, 3, padding=1),
                nn.GroupNorm(8, oc),
                nn.GELU(),
            )

        self.cnn = nn.Sequential(
            up(hidden, 256),   # 16 → 32
            up(256, 128),      # 32 → 64
            up(128, 64),       # 64 → 128
            up(64, 32),        # 128 → 256
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()       # → [0, 1] to match target range
        )

    def forward(self, tokens):       # tokens: [B, 1024, D]
        B = tokens.shape[0]
        x = self.proj(tokens)        # [B, 1024, hidden]
        # reshape: 1024 = 4 temporal × 16 × 16 spatial
        x = x.view(B, 4, 16, 16, -1).mean(dim=1)   # temporal pool → [B, 16, 16, hidden]
        x = x.permute(0, 3, 1, 2)                    # [B, hidden, 16, 16]
        return self.cnn(x)                             # [B, 3, 256, 256]


decoder = VJEPADecoder(hidden=HIDDEN_DIM).to(DEVICE)
n_params = sum(p.numel() for p in decoder.parameters())
print(f"\n[4] Decoder: {n_params:,} parameters")

# Quick sanity check
with torch.no_grad():
    dummy = torch.randn(2, 1024, 1024).to(DEVICE)
    out = decoder(dummy)
    assert out.shape == (2, 3, 256, 256), f"Unexpected shape: {out.shape}"
print("    Architecture check passed ✓")


# ── 5. Loss Function ────────────────────────────────────────────────────────
class ReconLoss(nn.Module):
    """MSE + gradient sharpness loss."""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.mse   = nn.MSELoss()
        self.alpha = alpha

    def gradient(self, x):
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dy, dx

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        py, px = self.gradient(pred)
        ty, tx = self.gradient(target)
        loss += self.alpha * (self.mse(py, ty) + self.mse(px, tx))
        return loss


criterion = ReconLoss(alpha=0.1)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── 6. Training Loop ────────────────────────────────────────────────────────
print(f"\n[5] Training for {EPOCHS} epochs...")
loss_history = []

for epoch in range(EPOCHS):
    decoder.train()
    epoch_loss = 0.0

    for emb, target in loader:
        emb    = emb.to(DEVICE)
        target = target.to(DEVICE)

        pred = decoder(emb)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg = epoch_loss / len(loader)
    loss_history.append(avg)

    if (epoch + 1) % 5 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"    Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg:.5f} | LR: {lr:.2e}")

torch.save(decoder.state_dict(), MODEL_SAVE)
print(f"\n    Decoder weights saved → {MODEL_SAVE}")

# Loss curve
plt.figure(figsize=(8, 3))
plt.plot(loss_history, linewidth=2, color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE + Gradient)")
plt.title("V-JEPA 2 Decoder — Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/decoder_loss_curve.png", dpi=120)
plt.show()


# ── 7. Reconstruction Visualisation ────────────────────────────────────────
print("\n[6] Visualising reconstructions...")
decoder.eval()
dataset_full = EmbeddingFrameDataset(DATASET_DIR)
indices = np.random.choice(len(dataset_full), min(5, len(dataset_full)), replace=False)

fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 7))

for col, idx in enumerate(indices):
    emb, target = dataset_full[idx]
    with torch.no_grad():
        pred = decoder(emb.unsqueeze(0).to(DEVICE))[0].cpu()

    orig = target.permute(1, 2, 0).numpy()
    recon = pred.clamp(0, 1).permute(1, 2, 0).numpy()
    psnr = -10 * np.log10(np.mean((orig - recon) ** 2) + 1e-8)

    axes[0, col].imshow(orig)
    axes[0, col].set_title(f"Original #{idx}", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(recon)
    axes[1, col].set_title(f"Decoded  PSNR={psnr:.1f}dB", fontsize=9)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Ground Truth",       fontsize=10, rotation=90)
axes[1, 0].set_ylabel("V-JEPA → Decoder",   fontsize=10, rotation=90)
plt.suptitle(
    "V-JEPA 2 Latent Space → Reconstructed Frame\n"
    "(CNN Decoder trained on MSE + Gradient Loss)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/tmp/decoder_reconstructions.png", dpi=120)
plt.show()

print("\n✓ Done!")
print("   Loss curve   → /tmp/decoder_loss_curve.png")
print("   Reconstructions → /tmp/decoder_reconstructions.png")
print(f"   Model weights   → {MODEL_SAVE}")
