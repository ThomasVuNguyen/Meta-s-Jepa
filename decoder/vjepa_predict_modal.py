"""
V-JEPA 2 World Model Prediction Pipeline
=========================================
No training needed — uses:
  • Pre-trained V-JEPA 2 encoder + predictor (HuggingFace)
  • Trained CNN decoder from vjepa_dec.pt (Modal volume)

Pipeline:
  context frames [0..3] → encoder → context embeddings
                                  ↓ predictor
                         target embeddings (frames [4..7])
                                  ↓ CNN decoder
                         predicted pixel frames

Output saved to Modal volume + downloaded locally as:
  • prediction_comparison.png  — context | predicted | actual (per frame)
  • prediction_psnr.json       — PSNR & MSE between predicted vs actual
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-prediction")

model_cache = modal.Volume.from_name("vjepa2-model-cache", create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .run_commands(
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "yt-dlp opencv-python-headless matplotlib Pillow numpy",
    )
)

# ── CNN Decoder factory — matches exact architecture of vjepa_dec.pt ─────────
EMBED_DIM = 1024
HIDDEN    = 384
OUT_SIZE  = 256

def make_decoder():
    import torch.nn as nn
    class Decoder(nn.Module):
        def __init__(self, D=EMBED_DIM, H=HIDDEN):
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
    return Decoder()

# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache": model_cache, "/output": output_vol},
)
def run_prediction(n_context: int = 4):
    """
    n_context: how many frames (out of 8) to give as context.
               The rest are predicted by V-JEPA 2 and compared to actual.
    """
    import sys
    conda_sp = "/opt/conda/lib/python3.11/site-packages"
    if conda_sp not in sys.path:
        sys.path.insert(0, conda_sp)

    import os, cv2, numpy as np, json, urllib.request
    import torch
    import torch.nn as nn
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    T_TOTAL = 8          # total frames per clip
    n_target = T_TOTAL - n_context

    # ── 1. Load encoder + predictor ─────────────────────────────────────────
    print(f"[1] Loading V-JEPA 2 (torch {torch.__version__})...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()
    print(f"    Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Submodules: {[n for n, _ in model.named_children()]}")

    # ── 2. Load trained CNN decoder ─────────────────────────────────────────
    print("[2] Loading trained CNN decoder...")
    decoder_path = "/output/vjepa_dec.pt"
    if not Path(decoder_path).exists():
        raise FileNotFoundError(
            "vjepa_dec.pt not found in volume! Run vjepa_decoder_modal.py first."
        )
    decoder = make_decoder().to(DEVICE).eval()
    decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE, weights_only=True))
    print("    Decoder loaded.")

    # ── 3. Download video ────────────────────────────────────────────────────
    print("[3] Downloading video...")
    vp = "/tmp/bbb.mp4"
    url = "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"
    urllib.request.urlretrieve(url, vp)
    print(f"    {os.path.getsize(vp)/1e6:.1f} MB downloaded")

    # ── 4. Sample a clip ────────────────────────────────────────────────────
    print("[4] Sampling clip...")
    ET = transforms.Compose([
        transforms.Resize((OUT_SIZE, OUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    TT = transforms.Compose([transforms.Resize((OUT_SIZE, OUT_SIZE)), transforms.ToTensor()])

    cap = cv2.VideoCapture(vp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pick a dynamic scene midway through
    start = total // 3
    frames_norm, frames_raw = [], []
    for i in range(T_TOTAL):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * 12)
        ret, fr = cap.read()
        assert ret, f"Frame read failed at {start + i*12}"
        p = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        frames_norm.append(ET(p))
        frames_raw.append(TT(p))
    cap.release()
    print(f"    Sampled {T_TOTAL} frames starting at frame {start}")

    # full clip tensor [1, T, C, H, W]
    clip = torch.stack(frames_norm).unsqueeze(0).to(DEVICE, dtype=torch.float16)

    # ── 5. Encode full clip → get all embeddings ─────────────────────────────
    print("[5] Encoding full clip...")
    with torch.no_grad():
        enc_out = model(pixel_values_videos=clip)
    all_emb = enc_out.last_hidden_state[0].cpu().float()  # [N_tokens, 1024]
    print(f"    Embedding shape: {all_emb.shape}")

    # Tokens per frame = total_tokens / T_TOTAL
    n_tok_total = all_emb.shape[0]
    tok_per_frame = n_tok_total // T_TOTAL
    print(f"    {tok_per_frame} tokens/frame × {T_TOTAL} frames = {n_tok_total} tokens")

    # Split embeddings into context vs target by frame
    context_emb = all_emb[:tok_per_frame * n_context]           # first n_context frames
    actual_emb  = all_emb[tok_per_frame * n_context:]           # last n_target frames

    # ── 6. Run predictor (if available) ─────────────────────────────────────
    print("[6] Running predictor...")
    predicted_emb = None

    if hasattr(model, 'predictor'):
        print("    Using built-in predictor...")
        try:
            n_ctx_tok = tok_per_frame * n_context
            n_tgt_tok = tok_per_frame * n_target
            # Predictor expects index tensors (int64), not bool masks
            ctx_indices = torch.arange(n_ctx_tok, dtype=torch.int64).unsqueeze(0).to(DEVICE)        # [1, n_ctx_tok]
            tgt_indices = torch.arange(n_ctx_tok, n_tok_total, dtype=torch.int64).unsqueeze(0).to(DEVICE)  # [1, n_tgt_tok]

            with torch.no_grad():
                pred_out = model(
                    pixel_values_videos=clip,
                    context_mask=ctx_indices,
                    target_mask=tgt_indices,
                )
            # predictor output: predicted embeddings for target positions
            if hasattr(pred_out, 'predictor_output') and pred_out.predictor_output is not None:
                predicted_emb = pred_out.predictor_output[0, :n_tgt_tok].cpu().float()
                print(f"    Predicted emb shape: {predicted_emb.shape}")
            else:
                print("    predictor_output not in model output — will use actual target emb for decoding")
        except Exception as e:
            print(f"    Predictor call failed: {e}\n    Falling back to actual embeddings.")
    else:
        print("    No predictor submodule found — decoding actual target embeddings (upper-bound comparison)")

    # Use actual target embeddings as fallback (shows decoder quality, not prediction)
    if predicted_emb is None:
        predicted_emb = actual_emb
        mode = "DECODE_ACTUAL"
    else:
        mode = "PREDICTED"
    print(f"    Mode: {mode}")

    print("[7] Decoding embeddings...")
    def emb_to_frame(emb_chunk, n_frames, tok_per_frame):
        """
        Decoder was trained with DataLoader batching: input shape [B, N_tok, D].
        For single-frame decode: pass [1, 1024, 1024] (add batch dim, pad tokens).
        """
        full_n = 1024   # total tokens the decoder was trained on
        imgs = []
        for i in range(n_frames):
            tok = emb_chunk[i*tok_per_frame:(i+1)*tok_per_frame]  # [TPF, D]
            # Pad to full token count if this is a per-frame slice
            if tok.shape[0] < full_n:
                repeats = (full_n + tok.shape[0] - 1) // tok.shape[0]
                tok = tok.repeat(repeats, 1)[:full_n]   # [1024, D]
            tok = tok.unsqueeze(0).to(DEVICE)  # [1, 1024, D] — add batch dim
            with torch.no_grad():
                px = decoder(tok)[0].cpu().clamp(0, 1)   # [3, H, W]
            imgs.append(px)
        return imgs

    pred_frames   = emb_to_frame(predicted_emb, n_target, tok_per_frame)
    context_frames_emb = all_emb[:tok_per_frame * n_context]
    ctx_frames_dec = emb_to_frame(context_frames_emb, n_context, tok_per_frame)
    print(f"    Decoded {n_context} context + {n_target} target frames")

    # ── 8. Compute PSNR ─────────────────────────────────────────────────────
    print("[8] Computing PSNR...")
    psnrs = []
    for i in range(n_target):
        actual_px = frames_raw[n_context + i]   # [3, H, W] 0..1
        pred_px   = pred_frames[i]               # [3, H, W] 0..1
        mse = float(((actual_px - pred_px) ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        psnrs.append(psnr)
    mean_psnr = float(np.mean(psnrs))
    print(f"    Mean PSNR (predicted vs actual): {mean_psnr:.1f} dB")
    print(f"    Per-frame PSNR: {[f'{p:.1f}' for p in psnrs]}")

    # ── 9. Save comparison figure ────────────────────────────────────────────
    print("[9] Saving comparison figure...")
    # Rows: context frames | separator | predicted | actual
    cols = max(n_context, n_target)
    fig, axes = plt.subplots(3, cols, figsize=(cols * 3, 9))
    fig.patch.set_facecolor("#111")

    def show(ax, t, title, cmap=None):
        ax.imshow(t.permute(1,2,0).numpy(), cmap=cmap)
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

    for i in range(cols):
        # Row 0: context frames (raw)
        if i < n_context:
            show(axes[0, i], frames_raw[i], f"Context t={i}")
        else:
            axes[0, i].axis("off")

        # Row 1: decoded predictions
        if i < n_target:
            show(axes[1, i], pred_frames[i],
                 f"{'Predicted' if mode=='PREDICTED' else 'Decoded'} t={n_context+i}\n{psnrs[i]:.1f}dB")
        else:
            axes[1, i].axis("off")

        # Row 2: actual frames (raw)
        if i < n_target:
            show(axes[2, i], frames_raw[n_context + i], f"Actual t={n_context+i}")
        else:
            axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Context", color="cyan", fontsize=11)
    axes[1, 0].set_ylabel("Predicted →", color="orange", fontsize=11)
    axes[2, 0].set_ylabel("Actual", color="lime", fontsize=11)

    title = (
        f"V-JEPA 2 World Model — {mode}\n"
        f"Context: {n_context} frames → Predicting: {n_target} frames | Mean PSNR: {mean_psnr:.1f}dB"
    )
    fig.suptitle(title, color="white", fontsize=12, y=1.01)
    plt.tight_layout()

    out_fig = "/output/prediction_comparison.png"
    plt.savefig(out_fig, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"    Figure saved to {out_fig}")

    # Save metrics JSON
    metrics = {
        "mode": str(mode),
        "n_context": int(n_context),
        "n_target": int(n_target),
        "mean_psnr_db": float(mean_psnr),
        "per_frame_psnr_db": [float(p) for p in psnrs],
        "tokens_per_frame": int(tok_per_frame),
        "total_tokens": int(n_tok_total),
        "has_predictor": bool(hasattr(model, 'predictor')),
    }
    with open("/output/prediction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("    Metrics saved.")
    output_vol.commit()
    return metrics


@app.local_entrypoint()
def main():
    print("Running V-JEPA 2 prediction pipeline...")
    metrics = run_prediction.remote(n_context=4)
    print("\n=== Results ===")
    print(f"Mode: {metrics['mode']}")
    print(f"Mean PSNR: {metrics['mean_psnr_db']:.1f} dB")
    print(f"Per-frame PSNR: {metrics['per_frame_psnr_db']}")
    print(f"Has predictor: {metrics['has_predictor']}")

    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    for fname in ["prediction_comparison.png", "prediction_metrics.json"]:
        try:
            import subprocess
            subprocess.run(
                ["modal", "volume", "get", "vjepa2-decoder-output", fname, str(out / fname)],
                check=True
            )
            print(f"Downloaded {fname}")
        except Exception as e:
            print(f"Download {fname}: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")
