"""
V-JEPA 2 Linear Probe — Robotics Capability Test
==================================================
Answers: "Are V-JEPA 2 embeddings useful as a robot perception backbone?"

Pipeline:
  1. Download Big Buck Bunny (varied scenes / motion)
  2. Auto-label frames with YOLOv8 (pretrained COCO) → bounding boxes
  3. Extract V-JEPA 2 embeddings for each labeled frame
  4. Train a linear probe on 80% of samples, test on 20%
  5. Compare against a random-feature baseline (same linear layer, random encoder)

Probe tasks (all from a single linear layer, no hidden units):
  A) Object XY position     → R² score  (where is it?)
  B) Object size (WH)       → R² score  (how big / far?)
  C) Motion direction       → accuracy  (moving left/right/up/down/still)
  D) Object class           → accuracy  (person / car / animal / other)

Outputs:
  probe_results.json   — R² and accuracy per task
  probe_summary.png    — bar chart comparing V-JEPA vs random baseline
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-probe")

model_cache = modal.Volume.from_name("vjepa2-model-cache", create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
video_cache = modal.Volume.from_name("vjepa2-video-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .run_commands(
        # Install headless OpenCV FIRST before ultralytics can pull the full one
        "/opt/conda/bin/pip install opencv-python-headless",
        # Install ultralytics without deps so it doesn't replace headless cv2
        "/opt/conda/bin/pip install --no-deps ultralytics",
        # Install everything else ultralytics needs (except opencv)
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scikit-learn "
        "scipy psutil py-cpuinfo requests pyyaml tqdm",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours for 50k
    volumes={
        "/cache": model_cache,
        "/output": output_vol,
        "/videos": video_cache,   # persist downloaded videos across runs
    },
)
def run_probe(n_frames: int = 10000):
    """
    n_frames: frames to sample PER video.
              10000 × 5 videos = ~50k labeled candidates on A10G.
              Videos are cached to a Modal volume — re-runs are fast.
    """
    import sys, os, cv2, json, urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conda_sp = "/opt/conda/lib/python3.11/site-packages"
    if conda_sp not in sys.path:
        sys.path.insert(0, conda_sp)

    DEVICE = "cuda"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"

    # ── 1. Load V-JEPA 2 ─────────────────────────────────────────────────────
    print("[1] Loading V-JEPA 2...")
    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()
    print(f"    {sum(p.numel() for p in vjepa.parameters()):,} params")

    # ── 2. Load YOLOv8 ───────────────────────────────────────────────────────
    print("[2] Loading YOLOv8...")
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")   # nano — fast, good enough for labeling

    # ── 3. Download / cache videos ────────────────────────────────────────
    print("[3] Ensuring videos are cached...")
    # All Blender Open Movie Project — CC-BY licensed
    videos = [
        ("https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4",
         "/videos/bbb.mp4"),
        ("https://archive.org/download/Sintel/sintel-2048-surround.mp4",
         "/videos/sintel.mp4"),
        ("https://archive.org/download/ElephantsDream/elephants_dream_1024_stereo.ogg",
         "/videos/elephants_dream.ogg"),
        ("https://archive.org/download/CosmosLaundromat/Cosmos_Laundromat_1080p.mp4",
         "/videos/cosmos.mp4"),
        ("https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4",
         "/videos/bbb2.mp4"),  # duplicate as 5th source if others fail
    ]
    video_paths = []
    for url, path in videos:
        try:
            if not Path(path).exists():
                print(f"    Downloading {Path(path).name}...")
                urllib.request.urlretrieve(url, path)
                video_cache.commit()  # persist to volume immediately
            sz = os.path.getsize(path) / 1e6
            if sz < 1:
                print(f"    Skipping {Path(path).name} (empty)")
                continue
            print(f"    {Path(path).name}: {sz:.1f} MB (cached)")
            video_paths.append(path)
        except Exception as e:
            print(f"    Skipping {Path(path).name}: {e}")

    # Deduplicate paths (bbb2 is a fallback)
    seen, unique_paths = set(), []
    for p in video_paths:
        key = os.path.getsize(p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)
    video_paths = unique_paths
    print(f"    {len(video_paths)} unique videos available")

    # ── 4. Sample frames evenly from all videos ───────────────────────────
    print(f"[4] Sampling {n_frames} frames per video ({len(video_paths)} videos)...")
    raw_frames = []   # PIL RGB, original size (for YOLO)
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices  = np.linspace(0, total_fr - 1, min(n_frames, total_fr - 1), dtype=int)
        vcount = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, fr = cap.read()
            if ret:
                raw_frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                vcount += 1
        cap.release()
        print(f"    {Path(vp).name}: {vcount} frames")
    print(f"    Total: {len(raw_frames)} frames")

    # ── 5. Auto-label with YOLO — GPU batched for speed ─────────────────────
    print("[5] Running YOLO detection (GPU batched)...")
    ANIMAL_IDS  = {15, 16, 17, 18, 19, 20, 21, 22, 23}
    VEHICLE_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
    PERSON_ID   = 0

    labels_xy   = []
    labels_wh   = []
    labels_cls  = []
    labeled_idx = []

    W, H = raw_frames[0].size
    YOLO_BATCH = 32  # GPU batched inference — much faster than frame-by-frame
    for batch_start in range(0, len(raw_frames), YOLO_BATCH):
        batch_imgs = raw_frames[batch_start:batch_start + YOLO_BATCH]
        results = yolo(batch_imgs, verbose=False, conf=0.35, device=0)  # device=0 = GPU
        for i, res in enumerate(results):
            global_i = batch_start + i
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best   = int(boxes.conf.argmax())
            cls_id = int(boxes.cls[best])
            xyxy   = boxes.xyxy[best].cpu().numpy()
            cx = float((xyxy[0] + xyxy[2]) / 2 / W)
            cy = float((xyxy[1] + xyxy[3]) / 2 / H)
            bw = float((xyxy[2] - xyxy[0]) / W)
            bh = float((xyxy[3] - xyxy[1]) / H)
            if   cls_id == PERSON_ID:   cls4 = 0
            elif cls_id in VEHICLE_IDS: cls4 = 1
            elif cls_id in ANIMAL_IDS:  cls4 = 2
            else:                       cls4 = 3
            labels_xy.append([cx, cy])
            labels_wh.append([bw, bh])
            labels_cls.append(cls4)
            labeled_idx.append(global_i)
        if batch_start % 2000 == 0:
            print(f"    YOLO: {batch_start}/{len(raw_frames)} done")

    n_labeled = len(labeled_idx)
    print(f"    {n_labeled}/{len(raw_frames)} frames with detections")
    if n_labeled < 50:
        raise RuntimeError("Too few detections — try a different video or lower conf threshold")

    labels_xy  = np.array(labels_xy,  dtype=np.float32)
    labels_wh  = np.array(labels_wh,  dtype=np.float32)
    labels_cls = np.array(labels_cls, dtype=np.int64)

    # Motion labels from consecutive detections
    motion_labels = []
    for k in range(len(labeled_idx)):
        if k == 0:
            motion_labels.append(4)  # still / unknown
        else:
            dx = labels_xy[k, 0] - labels_xy[k-1, 0]
            dy = labels_xy[k, 1] - labels_xy[k-1, 1]
            if abs(dx) < 0.02 and abs(dy) < 0.02:
                motion_labels.append(4)  # still
            elif abs(dx) > abs(dy):
                motion_labels.append(0 if dx > 0 else 1)  # right / left
            else:
                motion_labels.append(2 if dy > 0 else 3)  # down / up
    motion_labels = np.array(motion_labels, dtype=np.int64)

    # ── 6. Extract V-JEPA 2 embeddings ────────────────────────────────────────
    print(f"[6] Extracting V-JEPA 2 embeddings...")
    ET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embeddings = []
    batch_size = 32  # A10G 24GB VRAM — safely handles 32 frames at float16
    labeled_frames = [raw_frames[i] for i in labeled_idx]

    for start in range(0, len(labeled_frames), batch_size):
        batch_imgs = labeled_frames[start:start + batch_size]
        # Each "clip" = same frame repeated 8 times (model needs temporal dim)
        clips = []
        for img in batch_imgs:
            t = ET(img)                        # [C, H, W]
            clip = t.unsqueeze(0).repeat(8, 1, 1, 1)  # [8, C, H, W]
            clips.append(clip)
        clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)  # [B, 8, C, H, W]

        with torch.no_grad():
            out = vjepa(pixel_values_videos=clips)
            # last_hidden_state: [B, N_tokens, D] — mean-pool over tokens
            emb = out.last_hidden_state.mean(dim=1).cpu().float()  # [B, D]
        embeddings.append(emb)

        if (start // batch_size) % 5 == 0:
            print(f"    {start}/{len(labeled_frames)} frames encoded")

    embeddings = torch.cat(embeddings, dim=0).numpy()  # [N, 1024]
    print(f"    Embeddings shape: {embeddings.shape}")

    # ── 7. Random baseline embeddings ────────────────────────────────────────
    print("[7] Generating random baseline embeddings...")
    rng = np.random.RandomState(42)
    rand_embeddings = rng.randn(*embeddings.shape).astype(np.float32)

    # ── 8. Train + evaluate linear probes ────────────────────────────────────
    print("[8] Training linear probes...")
    N = len(embeddings)
    split = int(0.8 * N)
    perm  = rng.permutation(N)
    tr, te = perm[:split], perm[split:]

    results = {}

    def probe_regression(feat, label, name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat[tr])
        X_te = scaler.transform(feat[te])
        model = Ridge(alpha=1.0)
        model.fit(X_tr, label[tr])
        pred = model.predict(X_te)
        r2 = float(r2_score(label[te], pred))
        print(f"    {name:30s}: R² = {r2:.3f}")
        return r2

    def probe_classification(feat, label, name):
        # Filter out rare classes to avoid fitting errors
        unique, counts = np.unique(label, return_counts=True)
        valid_cls = unique[counts >= 3]
        mask_tr = np.isin(label[tr], valid_cls)
        mask_te = np.isin(label[te], valid_cls)
        if mask_tr.sum() < 10 or mask_te.sum() < 5:
            print(f"    {name:30s}: skipped (too few samples)")
            return None
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat[tr][mask_tr])
        X_te = scaler.transform(feat[te][mask_te])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, label[tr][mask_tr])
        pred = clf.predict(X_te)
        acc = float(accuracy_score(label[te][mask_te], pred))
        print(f"    {name:30s}: Acc = {acc:.3f}")
        return acc

    print("  --- V-JEPA 2 embeddings ---")
    results["vjepa_xy_r2"]     = probe_regression(embeddings,      labels_xy,     "  XY position  (V-JEPA)")
    results["vjepa_wh_r2"]     = probe_regression(embeddings,      labels_wh,     "  Size WH       (V-JEPA)")
    results["vjepa_cls_acc"]   = probe_classification(embeddings,  labels_cls,    "  Object class  (V-JEPA)")
    results["vjepa_mot_acc"]   = probe_classification(embeddings,  motion_labels, "  Motion dir    (V-JEPA)")

    print("  --- Random baseline ---")
    results["rand_xy_r2"]      = probe_regression(rand_embeddings,     labels_xy,     "  XY position  (random)")
    results["rand_wh_r2"]      = probe_regression(rand_embeddings,     labels_wh,     "  Size WH       (random)")
    results["rand_cls_acc"]    = probe_classification(rand_embeddings, labels_cls,    "  Object class  (random)")
    results["rand_mot_acc"]    = probe_classification(rand_embeddings, motion_labels, "  Motion dir    (random)")

    # ── 9. Summary chart ─────────────────────────────────────────────────────
    print("[9] Saving summary chart...")
    tasks  = ["XY Position\n(R²)", "Object Size\n(R²)", "Object Class\n(Acc)", "Motion Dir\n(Acc)"]
    vjepa_scores = [
        results["vjepa_xy_r2"],
        results["vjepa_wh_r2"],
        results["vjepa_cls_acc"] or 0,
        results["vjepa_mot_acc"] or 0,
    ]
    rand_scores = [
        results["rand_xy_r2"],
        results["rand_wh_r2"],
        results["rand_cls_acc"] or 0,
        results["rand_mot_acc"] or 0,
    ]

    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    bars1 = ax.bar(x - width/2, vjepa_scores, width, label="V-JEPA 2", color="#4fc3f7", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rand_scores,  width, label="Random baseline", color="#ef5350", edgecolor="white", linewidth=0.5, alpha=0.7)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", color="white", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", color="#ef5350", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, color="white", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=10)
    ax.set_title(
        "V-JEPA 2 Linear Probe — Can a single linear layer predict scene properties?\n"
        f"(n={N} frames, 80/20 train/test split)",
        color="white", fontsize=12
    )
    plt.tight_layout()
    plt.savefig("/output/probe_summary.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    results["n_labeled"] = int(n_labeled)
    results["n_total"]   = int(len(raw_frames))
    with open("/output/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    output_vol.commit()
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        if v is not None:
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return results


@app.local_entrypoint()
def main():
    print("Running V-JEPA 2 linear probe on Modal...")
    results = run_probe.remote(n_frames=10000)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    import subprocess
    for fname in ["probe_summary.png", "probe_results.json"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)],
                check=True
            )
            print(f"Downloaded {fname}")
        except Exception as e:
            print(f"Run manually: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")
