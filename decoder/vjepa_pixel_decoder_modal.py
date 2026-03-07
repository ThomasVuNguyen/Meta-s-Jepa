"""
V-JEPA 2 — Experiment 5: Latent Space Dreaming (Pixel Decoder)
================================================================
Trains a lightweight convolutional decoder  z (1024-d) → RGB (128×128)
on the DMControl frames already cached in the rollout volume.

Then uses the Phase-4-FT dynamics MLP to unroll an imagined trajectory
starting from a real observation, and decodes each imagined latent
back to pixel space to produce a "dream video".

Hypothesis: if V-JEPA's latent space is geometrically consistent, the
decoded sequences should look like smooth, physically plausible videos
of the reacher arm.  Blurriness / drift is expected but the arm shape
and reaching motion should be recognisable.

Architecture:
    z (1024) → MLP project → (512,4,4) → 5× ConvTranspose2d → (3,128,128)
    ~1.8M params, trains in ~20 min on A10G.

Compute estimate: ~45 min A10G total, ~$0.83
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-pixel-decoder")

model_cache = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf", "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scipy tqdm imageio[ffmpeg]",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/cache":    model_cache,
        "/output":   output_vol,
        "/rollouts": rollout_vol,
    },
)
def pixel_decoder_experiment(
    latent_dim:   int = 1024,
    px_out:       int = 128,
    epochs:       int = 60,
    batch_size:   int = 128,
    lr:           float = 2e-4,
    dream_steps:  int = 60,   # how many MLP-rollout steps to decode
    n_dreams:     int = 4,    # how many goal-conditioned dream sequences to make
    horizon:      int = 50,
    n_candidates: int = 256,
    n_elites:     int = 32,
    n_cem_iters:  int = 5,
):
    import os, json, time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path
    from PIL import Image as PILImage
    from torchvision import transforms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    DEVICE = "cuda"

    # ─── V-JEPA encoder (frozen) ────────────────────────────────────────
    print("[1] Loading V-JEPA 2 encoder (frozen)...")
    from transformers import AutoModel
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

    def embed_batch(frames_np, bs=32):
        embs = []
        for s in range(0, len(frames_np), bs):
            batch = frames_np[s:s + bs]
            clips = []
            for f in batch:
                img  = PILImage.fromarray(f)
                clip = ET(img).unsqueeze(0).repeat(8, 1, 1, 1)
                clips.append(clip)
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(embs, dim=0)

    def embed_single(frame_np):
        img  = PILImage.fromarray(frame_np)
        clip = ET(img).unsqueeze(0).repeat(8,1,1,1).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            out = vjepa(pixel_values_videos=clip)
            return out.last_hidden_state.mean(dim=1).squeeze(0).float()

    # ─── Pixel Decoder architecture ──────────────────────────────────────
    class PixelDecoder(nn.Module):
        """
        z (latent_dim) → project MLP → reshape (512, 4, 4)
                       → 5× ConvTranspose layers → (3, 128, 128)

        4 → 8 → 16 → 32 → 64 → 128  (6 doublings of spatial res)
        But we start at 4 so 5 ConvT gets us to 128.
        """
        def __init__(self, latent_dim=1024, out_size=128):
            super().__init__()
            # Project latent to spatial seed
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, 2048),
                nn.GELU(),
                nn.Linear(2048, 512 * 4 * 4),
            )
            # 4→8→16→32→64→128
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 8
                nn.BatchNorm2d(256), nn.GELU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16
                nn.BatchNorm2d(128), nn.GELU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32
                nn.BatchNorm2d(64),  nn.GELU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64
                nn.BatchNorm2d(32),  nn.GELU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 128
                nn.Sigmoid(),  # [0,1]
            )

        def forward(self, z):
            x = self.proj(z).view(-1, 512, 4, 4)
            return self.deconv(x)

    # ─── Load cached frames ──────────────────────────────────────────────
    print("[2] Loading cached DMControl frames...")
    all_frames = []
    for key in ["reacher_easy_goal", "reacher_easy"]:
        d = Path(f"/rollouts/{key}")
        if not d.exists():
            print(f"    [SKIP] {key}"); continue
        frames = np.load(str(d / "frames.npy"))
        all_frames.append(frames)
        print(f"    {key}: {len(frames)} frames  shape={frames.shape}")

    frames_np = np.concatenate(all_frames, axis=0)
    print(f"    Total: {len(frames_np)} frames")

    # Subsample to keep memory reasonable (10k is enough for decoder)
    if len(frames_np) > 12000:
        idx = np.random.RandomState(7).choice(len(frames_np), 12000, replace=False)
        frames_np = frames_np[idx]
        print(f"    Subsampled to {len(frames_np)} frames")

    # ─── Embed all frames ─────────────────────────────────────────────
    print("[3] Embedding frames with V-JEPA 2...")
    t0 = time.time()
    Z = embed_batch(frames_np, bs=32)  # (N, 1024)
    print(f"    Embedded {len(Z)} frames in {(time.time()-t0)/60:.1f} min")

    # Prepare pixel targets: resize to 128×128, float [0,1]
    print("[4] Preparing pixel targets (128×128)...")
    pixel_targets = []
    for f in frames_np:
        img = PILImage.fromarray(f).resize((px_out, px_out), PILImage.BILINEAR)
        pixel_targets.append(np.array(img, dtype=np.float32) / 255.0)
    pixel_targets = np.stack(pixel_targets)                     # (N, H, W, 3)
    pixel_targets = pixel_targets.transpose(0, 3, 1, 2)        # (N, 3, H, W)
    X_px = torch.tensor(pixel_targets, dtype=torch.float32)

    # ─── Train decoder ───────────────────────────────────────────────────
    print(f"[5] Training PixelDecoder ({epochs} epochs, batch={batch_size})...")
    N = len(Z)
    perm  = np.random.RandomState(42).permutation(N)
    split = int(0.9 * N)
    tr, te = perm[:split], perm[split:]

    train_ds = TensorDataset(Z[tr], X_px[tr])
    test_ds  = TensorDataset(Z[te], X_px[te])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    decoder   = PixelDecoder(latent_dim, px_out).to(DEVICE)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Perceptual-style loss: L2 + L1 (L1 sharpens blurry decoder outputs)
    def recon_loss(pred, target):
        return F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)

    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        decoder.train()
        ep_loss = 0.0
        for z_b, px_b in train_dl:
            z_b  = z_b.to(DEVICE)
            px_b = px_b.to(DEVICE)
            optimizer.zero_grad()
            pred = decoder(z_b)
            loss = recon_loss(pred, px_b)
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(z_b)
        scheduler.step()

        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_b, px_b in test_dl:
                z_b  = z_b.to(DEVICE)
                px_b = px_b.to(DEVICE)
                val_loss += recon_loss(decoder(z_b), px_b).item() * len(z_b)
        train_losses.append(ep_loss / len(train_ds))
        val_losses.append(val_loss / len(test_ds))
        if epoch % 10 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"    epoch {epoch}/{epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  [{elapsed:.1f}min]")

    # Save decoder
    torch.save({
        "model_state": decoder.state_dict(),
        "latent_dim": latent_dim, "out_size": px_out,
        "final_val_loss": val_losses[-1],
    }, "/output/pixel_decoder.pt")
    output_vol.commit()
    print(f"    ✓ Decoder saved (val_loss={val_losses[-1]:.4f})")

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    ax.plot(range(1, epochs+1), train_losses, color="#4fc3f7", label="Train")
    ax.plot(range(1, epochs+1), val_losses,   color="#ffa726", label="Val")
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("L2+0.5×L1 loss", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white")
    ax.set_title("PixelDecoder Training Curve", color="white")
    plt.tight_layout()
    plt.savefig("/output/pixel_decoder_training.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ─── Reconstruction quality: ground-truth vs decoded ─────────────────
    print("[6] Visualising reconstructions (ground-truth vs decoded)...")
    decoder.eval()
    n_show = 6
    sample_idx = np.random.RandomState(0).choice(len(Z[te]), n_show, replace=False)
    z_sample  = Z[te][sample_idx].to(DEVICE)
    px_sample = X_px[te][sample_idx]   # (n_show, 3, H, W)
    with torch.no_grad():
        recon = decoder(z_sample).cpu()  # (n_show, 3, H, W)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))
    fig.patch.set_facecolor("#111")
    for ax in axes.flat: ax.axis("off")
    for i in range(n_show):
        gt  = px_sample[i].permute(1, 2, 0).numpy()
        dec = recon[i].permute(1, 2, 0).numpy().clip(0, 1)
        axes[0, i].imshow(gt);  axes[0, i].set_title("GT",  color="white", fontsize=8)
        axes[1, i].imshow(dec); axes[1, i].set_title("Dec", color="white", fontsize=8)
    fig.suptitle("Ground-Truth vs Decoded (from V-JEPA latent)", color="white", fontsize=11)
    plt.tight_layout()
    plt.savefig("/output/pixel_decoder_recon.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    output_vol.commit()

    # ─── Load dynamics MLP (Phase 4 FT checkpoint) ────────────────────────
    print("[7] Loading Phase 4→R2 dynamics checkpoint for dreaming...")
    # Prefer R2 (best so far), fall back to Phase 4 FT
    z_dim = 1024; a_pad_dim = 6; hidden = 512; n_layers = 3

    class DynamicsMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []; in_d = z_dim + a_pad_dim
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU()]
                in_d = hidden
            layers.append(nn.Linear(in_d, z_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1))

    def pad_action(a, pad_to=6):
        return np.concatenate([a, np.zeros(pad_to - len(a), dtype=np.float32)])

    r2_candidates = sorted(Path("/output").glob("dynamics_mlp_dyna_r2*.pt"))
    ft_candidates = sorted(Path("/output").glob("dynamics_mlp_ft*.pt"), reverse=True)
    ckpt_path = (r2_candidates[-1] if r2_candidates else
                 ft_candidates[0]  if ft_candidates else None)
    if ckpt_path is None:
        raise FileNotFoundError("No dynamics checkpoint found!")
    print(f"    Using: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    dynamics = DynamicsMLP().to(DEVICE)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()

    # ─── Dream sequences ──────────────────────────────────────────────────
    print(f"[8] Generating {n_dreams} dream sequences ({dream_steps} steps each)...")
    from dm_control import suite

    rng = np.random.RandomState(77)
    all_dream_frames = []  # list of (dream_steps, H, W, 3) arrays

    for dream_id in range(n_dreams):
        print(f"    Dream {dream_id+1}/{n_dreams}...")
        env      = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(0, 10000)})
        env_goal = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(10000, 20000)})
        aspec    = env.action_spec()

        env.reset(); env_goal.reset()
        for _ in range(rng.randint(5, 30)):
            env_goal.step(rng.uniform(-1, 1, size=2))

        goal_frame = env_goal.physics.render(height=256, width=256, camera_id=0)
        z_goal     = embed_single(goal_frame).to(DEVICE)
        start_frame = env.physics.render(height=256, width=256, camera_id=0)
        z_curr      = embed_single(start_frame).to(DEVICE)

        # Run CEM MPC for dream_steps, decode each imagined latent
        dream_frames_decoded = []  # decoded latents
        real_frames_side     = []  # real env frames for side-by-side

        for step in range(dream_steps):
            # Decode current latent to pixel
            with torch.no_grad():
                decoded = decoder(z_curr.unsqueeze(0)).squeeze(0).cpu().numpy()  # (3, H, W)
                decoded = (decoded.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            dream_frames_decoded.append(decoded)

            # Store real frame for comparison
            real_frame = env.physics.render(height=px_out, width=px_out, camera_id=0)
            real_frames_side.append(real_frame)

            # CEM planning step
            N_c = n_candidates; K = n_elites; I = n_cem_iters
            mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
            sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
            z_s = z_curr.unsqueeze(0).expand(N_c, -1)
            z_g = z_goal.unsqueeze(0).expand(N_c, -1)
            for _ in range(I):
                eps      = rng.randn(N_c, horizon, a_pad_dim).astype(np.float32)
                act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
                act_t    = torch.tensor(act_seqs, device=DEVICE)
                z_c = z_s.clone()
                with torch.no_grad():
                    for t in range(horizon):
                        z_c = dynamics(z_c, act_t[:, t, :])
                costs     = ((z_c - z_g)**2).sum(dim=-1).cpu().numpy()
                elite_idx = np.argsort(costs)[:K]
                mu  = act_seqs[elite_idx].mean(axis=0)
                sig = act_seqs[elite_idx].std(axis=0) + 1e-6

            # Execute first action in real env
            a_exec = np.clip(mu[0, :2], aspec.minimum, aspec.maximum)
            env.step(a_exec)

            # Advance latent with imagined action (for next step decode)
            a_pad = torch.tensor(pad_action(a_exec), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                z_curr = dynamics(z_curr.unsqueeze(0), a_pad).squeeze(0)

        all_dream_frames.append((real_frames_side, dream_frames_decoded))

    # ─── Create side-by-side videos ──────────────────────────────────────
    print("[9] Saving dream videos and montage...")
    for dream_id, (real_seq, dream_seq) in enumerate(all_dream_frames):
        video_path = f"/output/dream_{dream_id:02d}.mp4"
        frames_out = []
        for r, d in zip(real_seq, dream_seq):
            # side-by-side: real | black divider | dream
            divider = np.zeros((px_out, 4, 3), dtype=np.uint8)
            row = np.concatenate([r, divider, d], axis=1)
            frames_out.append(row)
        writer = imageio.get_writer(video_path, fps=10, codec="libx264", quality=7)
        for f in frames_out:
            writer.append_data(f)
        writer.close()
        print(f"    ✓ dream_{dream_id:02d}.mp4")

    # Static montage: first frame of each dream
    fig, axes = plt.subplots(2, n_dreams, figsize=(n_dreams * 3, 7))
    fig.patch.set_facecolor("#111")
    for ax in axes.flat: ax.axis("off")
    for i, (real_seq, dream_seq) in enumerate(all_dream_frames):
        # Middle frame (step 30) for more interesting view
        mid = min(30, len(real_seq) - 1)
        axes[0, i].imshow(real_seq[mid])
        axes[0, i].set_title(f"Dream {i+1} — Real t=30", color="white", fontsize=8)
        axes[1, i].imshow(dream_seq[mid])
        axes[1, i].set_title(f"Dream {i+1} — Decoded t=30", color="white", fontsize=8)
    fig.suptitle(
        "V-JEPA Latent Space Dreaming\n"
        "Top: real environment frames  |  Bottom: MLP-imagined latents decoded to pixels",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("/output/dream_montage.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    output_vol.commit()

    results = {
        "experiment": 5,
        "decoder_params": sum(p.numel() for p in decoder.parameters()),
        "n_train_frames": len(train_ds),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "n_dreams": n_dreams,
        "dream_steps": dream_steps,
        "dynamics_ckpt": ckpt_path.name,
    }
    with open("/output/pixel_decoder_results.json", "w") as f:
        import json; json.dump(results, f, indent=2)
    output_vol.commit()

    print("\n=== EXPERIMENT 5 COMPLETE ===")
    print(f"  Decoder val_loss: {val_losses[-1]:.4f}")
    print(f"  Decoder params:   {results['decoder_params']:,}")
    print(f"  Dreams:           {n_dreams} × {dream_steps} steps")
    print(f"  Dynamics from:    {ckpt_path.name}")
    return results


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Experiment 5: Latent Space Dreaming")
    print("=" * 60)

    results = pixel_decoder_experiment.remote()
    print(json.dumps(results, indent=2))

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fname in [
        "pixel_decoder_results.json",
        "pixel_decoder_training.png",
        "pixel_decoder_recon.png",
        "dream_montage.png",
        "dream_00.mp4", "dream_01.mp4", "dream_02.mp4", "dream_03.mp4",
    ]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)],
                check=True,
            )
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  skip {fname}: {e}")
    print("Done!")
