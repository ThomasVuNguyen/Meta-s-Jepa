"""
vjepa_decoder_modal.py
======================
Run V-JEPA 2 decoder training on Modal cloud GPU.

Usage:
  pip install modal
  modal setup          # one-time login
  modal run vjepa_decoder_modal.py

Results (loss curve + reconstructions) are saved locally to ./decoder_output/
"""

import modal

# ── Modal App definition ────────────────────────────────────────────────────
app = modal.App("vjepa-decoder")

# Container image: starts from CUDA-enabled base, installs our deps
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime",
    )
    .run_commands(
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors yt-dlp opencv-python-headless matplotlib Pillow numpy",
    )
)

# Volume to persist downloaded model weights across runs (saves ~3 min each time)
model_cache = modal.Volume.from_name("vjepa2-weights", create_if_missing=True)

# Output volume: results are written here, then downloaded locally
output_vol = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",                   # change to "A10G" for ~2x faster training
    timeout=1800,               # 30 min max
    volumes={
        "/cache": model_cache,
        "/output": output_vol,
    },
)
def train_decoder(epochs: int = 25, hidden_dim: int = 384):
    """Full pipeline: load encoder → build dataset → train decoder → save plots."""
    import sys
    # Ensure conda's site-packages are first on path (pytorch/pytorch docker image)
    conda_sp = "/opt/conda/lib/python3.10/site-packages"
    if conda_sp not in sys.path:
        sys.path.insert(0, conda_sp)

    import os, subprocess, cv2, numpy as np
    import torch
    import torch.nn as nn
    # torch must be imported BEFORE transformers can pass its backend check
    assert torch.cuda.is_available(), f"CUDA not available, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from pathlib import Path
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    DDIR = Path("/tmp/jd"); DDIR.mkdir(exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

    # ── 1. Load encoder ─────────────────────────────────────────────────────
    print(f"[1] Loading V-JEPA 2 encoder (torch {torch.__version__})...")
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    )
    encoder = encoder.to(DEVICE, dtype=torch.float16).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"    Encoder loaded — {sum(p.numel() for p in encoder.parameters()):,} params")

    # ── 2. Build dataset ─────────────────────────────────────────────────────
    ET = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    TT = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    print("[2] Downloading video...")
    vp = "/tmp/bbb.mp4"
    import urllib.request
    url = "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"
    print(f"    Fetching {url} ...")
    urllib.request.urlretrieve(url, vp)
    fsize = os.path.getsize(vp) if os.path.exists(vp) else 0
    print(f"    Video downloaded: {fsize/1e6:.1f} MB")
    if fsize < 10000:
        raise RuntimeError(f"Video download failed ({fsize} bytes)")

    print("[3] Encoding clips...")
    cap = cv2.VideoCapture(vp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    print(f"    Video: {total} frames @ {fps:.1f}fps")
    starts = np.linspace(0, max(0, total - 60), 40, dtype=int)
    n = 0
    for ci, s in enumerate(starts):
        fe, ft = [], []
        for t in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s + t * 4))
            ret, fr = cap.read()
            if not ret: break
            p = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            fe.append(ET(p)); ft.append(TT(p))
        if len(fe) < 8: continue
        # Model takes [B, T, C, H, W] — stack frames, add batch dim
        vid = torch.stack(fe).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        # vid shape: [1, 8, 3, 256, 256] = [B, T, C, H, W] ✓
        with torch.no_grad():
            emb = encoder(pixel_values_videos=vid).last_hidden_state[0].cpu().float()
        torch.save({"e": emb, "t": ft[3]}, DDIR / f"p{ci:04d}.pt")
        n += 1
    cap.release()
    print(f"    {n} training pairs saved")

    # ── 3. Dataset + Decoder ─────────────────────────────────────────────────
    class DS(Dataset):
        def __init__(self, d): self.f = sorted(Path(d).glob("p*.pt"))
        def __len__(self): return len(self.f)
        def __getitem__(self, i):
            d = torch.load(self.f[i]); return d["e"], d["t"]

    loader = DataLoader(DS(DDIR), batch_size=4, shuffle=True)

    class Decoder(nn.Module):
        def __init__(self, D=1024, H=hidden_dim):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(D, H), nn.LayerNorm(H), nn.GELU())
            def up(i, o): return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
            self.cnn = nn.Sequential(
                up(H, 256), up(256, 128), up(128, 64), up(64, 32),
                nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())
        def forward(self, x):
            B = x.shape[0]
            x = self.proj(x).view(B, 4, 16, 16, -1).mean(1).permute(0, 3, 1, 2)
            return self.cnn(x)

    dec = Decoder().to(DEVICE)
    print(f"[4] Decoder: {sum(p.numel() for p in dec.parameters()):,} params")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    def loss_fn(p, t):
        l = mse(p, t)
        l += 0.1 * (mse(p[:,:,1:]-p[:,:,:-1], t[:,:,1:]-t[:,:,:-1]) +
                    mse(p[:,:,:,1:]-p[:,:,:,:-1], t[:,:,:,1:]-t[:,:,:,:-1]))
        return l

    hist = []
    print(f"[5] Training {epochs} epochs...")
    for ep in range(epochs):
        dec.train(); el = 0
        for e, t in loader:
            e, t = e.to(DEVICE), t.to(DEVICE)
            p = dec(e); l = loss_fn(p, t)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); el += l.item()
        sched.step(); hist.append(el / len(loader))
        if (ep + 1) % 5 == 0:
            print(f"    Epoch {ep+1:2d}/{epochs} | loss={hist[-1]:.5f}")

    # Save weights to output volume
    torch.save(dec.state_dict(), "/output/vjepa_dec.pt")
    print("    Weights saved to /output/vjepa_dec.pt")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 2.5))
    plt.plot(hist, lw=2, color="steelblue"); plt.grid(alpha=0.3)
    plt.title("V-JEPA 2 Decoder — Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig("/output/decoder_loss.png", dpi=120)
    plt.close()

    dec.eval()
    ds = DS(DDIR)
    idxs = np.random.choice(len(ds), 5, replace=False)
    fig, ax = plt.subplots(2, 5, figsize=(18, 7))
    psnrs = []
    for col, idx in enumerate(idxs):
        e, t = ds[idx]
        with torch.no_grad():
            pred = dec(e.unsqueeze(0).to(DEVICE))[0].cpu()
        orig = t.permute(1, 2, 0).numpy()
        rec = pred.clamp(0, 1).permute(1, 2, 0).numpy()
        psnr = -10 * np.log10(((orig - rec)**2).mean() + 1e-8)
        psnrs.append(psnr)
        ax[0, col].imshow(orig); ax[0, col].set_title(f"Original #{idx}", fontsize=8); ax[0, col].axis("off")
        ax[1, col].imshow(rec);  ax[1, col].set_title(f"PSNR {psnr:.1f}dB", fontsize=8); ax[1, col].axis("off")
    ax[0, 0].set_ylabel("Ground Truth", fontsize=10)
    ax[1, 0].set_ylabel("V-JEPA 2 → Decoder", fontsize=10)
    plt.suptitle("V-JEPA 2 Latent → Reconstructed Frames", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/output/decoder_reconstructions.png", dpi=120)
    plt.close()

    result = {
        "final_loss": hist[-1],
        "mean_psnr": float(np.mean(psnrs)),
        "loss_history": hist,
    }
    print(f"\n✓ Done! Final loss: {hist[-1]:.5f} | Mean PSNR: {np.mean(psnrs):.1f}dB")
    return result


@app.local_entrypoint()
def main():
    import os
    from pathlib import Path

    print("Submitting job to Modal...")
    result = train_decoder.remote(epochs=25, hidden_dim=384)

    print(f"\nResult: {result}")

    # Download output files to local machine
    out_dir = Path("./decoder_output")
    out_dir.mkdir(exist_ok=True)

    output_vol = modal.Volume.from_name("vjepa2-decoder-output")
    for filename in ["vjepa_dec.pt", "decoder_loss.png", "decoder_reconstructions.png"]:
        remote_path = f"/{filename}"
        local_path = out_dir / filename
        try:
            with output_vol.batch_download() as download:
                download(remote_path, str(local_path))
            print(f"Downloaded: {local_path}")
        except Exception as e:
            print(f"Note: Could not auto-download {filename}: {e}")
            print(f"  → Run: modal volume get vjepa2-decoder-output {filename} {local_path}")

    print(f"\nOutput files in: {out_dir.absolute()}")
    print("  decoder_loss.png          — training loss curve")
    print("  decoder_reconstructions.png — original vs decoded frames")
    print("  vjepa_dec.pt              — trained decoder weights")
