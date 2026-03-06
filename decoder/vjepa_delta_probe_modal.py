"""
V-JEPA 2 Temporal Delta Probe
==============================
Phase 1 of Meta-s-Jepa research.

Hypothesis: Motion direction is encoded in Δz = z[t+1] - z[t] (temporal
difference of embeddings) rather than in a single frame embedding z[t].

If true: linear probe on Δz → motion direction should significantly
outperform probe on raw z[t] (current baseline: 63.6%).

This directly validates the core assumption behind the lightweight
latent dynamics model (MLP: z_t + a_t → z_{t+1}) for CEM planning.

Comparison:
  A) Raw z probe      (baseline, reproduced from vjepa_probe_modal.py)
  B) Δz probe         (temporal delta — the new test)
  C) [z_t ⊕ z_{t+1}] probe  (paired frames, upper-bound reference)
  D) Random baseline  (sanity check)

Outputs (saved to vjepa2-decoder-output volume):
  delta_probe_results.json     — all accuracy scores
  delta_probe_summary.png      — bar chart comparison
  embeddings_cache.npz         — cached embeddings for future probes (free reruns)
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-delta-probe")

model_cache = modal.Volume.from_name("vjepa2-model-cache",   create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
video_cache = modal.Volume.from_name("vjepa2-video-cache",   create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .run_commands(
        "/opt/conda/bin/pip install opencv-python-headless",
        "/opt/conda/bin/pip install --no-deps ultralytics",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scikit-learn "
        "scipy psutil py-cpuinfo requests pyyaml tqdm",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,
    volumes={
        "/cache":   model_cache,
        "/output":  output_vol,
        "/videos":  video_cache,
    },
)
def run_delta_probe(n_frames: int = 10000, force_reextract: bool = False):
    """
    n_frames:        frames sampled per video (same as original probe for fair comparison)
    force_reextract: if True, ignore cached embeddings and re-run extraction
    """
    import sys, os, cv2, json, urllib.request
    import numpy as np
    import torch
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
    os.environ["YOLO_CONFIG_DIR"]    = "/tmp/yolo"

    CACHE_PATH = Path("/output/embeddings_cache.npz")

    # ── 1. Try loading cached embeddings ─────────────────────────────────────
    embeddings      = None
    labeled_idx     = None
    labels_xy       = None
    labels_wh       = None
    labels_cls      = None
    motion_labels   = None
    n_total         = None

    if CACHE_PATH.exists() and not force_reextract:
        print("[1] Loading cached embeddings from volume...")
        try:
            data = np.load(CACHE_PATH, allow_pickle=True)
            embeddings    = data["embeddings"]
            labeled_idx   = data["labeled_idx"]
            labels_xy     = data["labels_xy"]
            labels_wh     = data["labels_wh"]
            labels_cls    = data["labels_cls"]
            motion_labels = data["motion_labels"]
            n_total       = int(data["n_total"])
            print(f"    Loaded {len(embeddings)} cached embeddings — skipping extraction!")
        except Exception as e:
            print(f"    Cache load failed ({e}), re-extracting...")
            embeddings = None

    # ── 2. Full extraction pipeline (only if cache miss) ─────────────────────
    if embeddings is None:
        print("[1] Cache miss — running full extraction pipeline...")

        # Load V-JEPA 2
        print("[2] Loading V-JEPA 2...")
        from transformers import AutoModel
        vjepa = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            trust_remote_code=True,
            cache_dir="/cache/hf",
        ).to(DEVICE, dtype=torch.float16).eval()
        print(f"    {sum(p.numel() for p in vjepa.parameters()):,} params")

        # Load YOLOv8
        print("[3] Loading YOLOv8...")
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")

        # Download / cache videos
        print("[4] Ensuring videos are cached...")
        videos = [
            ("https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4",
             "/videos/bbb.mp4"),
            ("https://archive.org/download/Sintel/sintel-2048-surround.mp4",
             "/videos/sintel.mp4"),
            ("https://archive.org/download/CosmosLaundromat/Cosmos_Laundromat_1080p.mp4",
             "/videos/cosmos.mp4"),
        ]
        video_paths = []
        for url, path in videos:
            try:
                if not Path(path).exists():
                    print(f"    Downloading {Path(path).name}...")
                    urllib.request.urlretrieve(url, path)
                    video_cache.commit()
                sz = os.path.getsize(path) / 1e6
                if sz < 1:
                    continue
                print(f"    {Path(path).name}: {sz:.1f} MB")
                video_paths.append(path)
            except Exception as e:
                print(f"    Skipping {Path(path).name}: {e}")

        # Sample frames
        print(f"[5] Sampling {n_frames} frames per video...")
        raw_frames = []
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
        n_total = len(raw_frames)
        print(f"    Total: {n_total} frames")

        # YOLO labeling
        print("[6] Running YOLO detection...")
        ANIMAL_IDS  = {15, 16, 17, 18, 19, 20, 21, 22, 23}
        VEHICLE_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
        PERSON_ID   = 0

        _labels_xy  = []
        _labels_wh  = []
        _labels_cls = []
        _labeled_idx = []

        W, H = raw_frames[0].size
        YOLO_BATCH = 32
        for batch_start in range(0, len(raw_frames), YOLO_BATCH):
            batch_imgs = raw_frames[batch_start:batch_start + YOLO_BATCH]
            results    = yolo(batch_imgs, verbose=False, conf=0.35, device=0)
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
                _labels_xy.append([cx, cy])
                _labels_wh.append([bw, bh])
                _labels_cls.append(cls4)
                _labeled_idx.append(global_i)
            if batch_start % 2000 == 0:
                print(f"    YOLO: {batch_start}/{n_total} done")

        labels_xy   = np.array(_labels_xy,   dtype=np.float32)
        labels_wh   = np.array(_labels_wh,   dtype=np.float32)
        labels_cls  = np.array(_labels_cls,  dtype=np.int64)
        labeled_idx = np.array(_labeled_idx, dtype=np.int64)
        print(f"    {len(labeled_idx)}/{n_total} frames with detections")

        # Motion labels
        _motion = []
        for k in range(len(labeled_idx)):
            if k == 0:
                _motion.append(4)
            else:
                dx = labels_xy[k, 0] - labels_xy[k-1, 0]
                dy = labels_xy[k, 1] - labels_xy[k-1, 1]
                if abs(dx) < 0.02 and abs(dy) < 0.02:
                    _motion.append(4)
                elif abs(dx) > abs(dy):
                    _motion.append(0 if dx > 0 else 1)
                else:
                    _motion.append(2 if dy > 0 else 3)
        motion_labels = np.array(_motion, dtype=np.int64)

        # Extract V-JEPA 2 embeddings
        print(f"[7] Extracting V-JEPA 2 embeddings for {len(labeled_idx)} frames...")
        ET = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        _embeddings = []
        batch_size  = 32
        labeled_frames = [raw_frames[i] for i in labeled_idx]

        for start in range(0, len(labeled_frames), batch_size):
            batch_imgs = labeled_frames[start:start + batch_size]
            clips = []
            for img in batch_imgs:
                t = ET(img)
                clip = t.unsqueeze(0).repeat(8, 1, 1, 1)  # [8, C, H, W]
                clips.append(clip)
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)

            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips)
                emb = out.last_hidden_state.mean(dim=1).cpu().float()
            _embeddings.append(emb)

            if (start // batch_size) % 5 == 0:
                print(f"    {start}/{len(labeled_frames)} encoded")

        embeddings = torch.cat(_embeddings, dim=0).numpy()
        print(f"    Embeddings shape: {embeddings.shape}")

        # Save to volume for future runs
        print("[8] Caching embeddings to volume...")
        np.savez_compressed(
            str(CACHE_PATH),
            embeddings=embeddings,
            labeled_idx=labeled_idx,
            labels_xy=labels_xy,
            labels_wh=labels_wh,
            labels_cls=labels_cls,
            motion_labels=motion_labels,
            n_total=np.array([n_total]),
        )
        output_vol.commit()
        print(f"    Saved embeddings_cache.npz ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")

    n_labeled = len(embeddings)

    # ── 3. Build temporal delta embeddings ───────────────────────────────────
    print("[9] Building temporal delta embeddings (Δz = z[t+1] - z[t])...")
    #
    # Key requirement: only compute Δz for CONSECUTIVE pairs of LABELED frames.
    # labeled_idx[k] tells us the original video frame index.
    # If labeled_idx[k+1] != labeled_idx[k] + 1, the pair is from non-adjacent
    # frames (object disappeared in frames between), so skip it.
    #
    _delta_emb    = []
    _delta_motion = []
    _pair_raw_t   = []  # z[t] for paired comparison
    _pair_raw_t1  = []  # z[t+1] for paired comparison
    _concat_emb   = []  # [z_t ⊕ z_t+1] for upper-bound reference

    skip_count = 0
    for k in range(len(labeled_idx) - 1):
        # Only use consecutive video frames (gap of 1)
        if labeled_idx[k+1] != labeled_idx[k] + 1:
            skip_count += 1
            continue
        # Skip "still" motion or first-frame unknown for cleaner signal
        # (keep all classes including still for full evaluation)
        delta = embeddings[k+1] - embeddings[k]
        _delta_emb.append(delta)
        _delta_motion.append(motion_labels[k+1])
        _pair_raw_t.append(embeddings[k])
        _pair_raw_t1.append(embeddings[k+1])
        _concat_emb.append(np.concatenate([embeddings[k], embeddings[k+1]]))

    delta_embeddings = np.array(_delta_emb,    dtype=np.float32)  # [M, 1024]
    delta_motion_lbl = np.array(_delta_motion, dtype=np.int64)    # [M]
    concat_embeddings= np.array(_concat_emb,   dtype=np.float32)  # [M, 2048]

    n_pairs = len(delta_embeddings)
    print(f"    {n_pairs} consecutive pairs (skipped {skip_count} non-adjacent)")
    print(f"    Δz shape: {delta_embeddings.shape}")

    # Motion class distribution
    unique, counts = np.unique(delta_motion_lbl, return_counts=True)
    labels_map = {0: "right", 1: "left", 2: "down", 3: "up", 4: "still"}
    for u, c in zip(unique, counts):
        print(f"    Class {u} ({labels_map.get(u,'?')}): {c} samples ({100*c/n_pairs:.1f}%)")

    # ── 4. Linear probe helpers ───────────────────────────────────────────────
    rng   = np.random.RandomState(42)
    perm  = rng.permutation(n_pairs)
    split = int(0.8 * n_pairs)
    tr, te = perm[:split], perm[split:]

    # For fair comparison: also probe raw z using only the SAME paired frames
    raw_t_embeddings = np.array(_pair_raw_t, dtype=np.float32)   # [M, 1024]

    def probe_classification(feat, label, name, tr_idx, te_idx):
        unique, counts = np.unique(label, return_counts=True)
        valid_cls = unique[counts >= 3]
        mask_tr = np.isin(label[tr_idx], valid_cls)
        mask_te = np.isin(label[te_idx], valid_cls)
        if mask_tr.sum() < 10 or mask_te.sum() < 5:
            print(f"    {name:45s}: skipped (too few samples)")
            return None
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat[tr_idx][mask_tr])
        X_te = scaler.transform(feat[te_idx][mask_te])
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(X_tr, label[tr_idx][mask_tr])
        pred = clf.predict(X_te)
        acc  = float(accuracy_score(label[te_idx][mask_te], pred))
        chance = 1.0 / len(valid_cls)
        lift   = (acc - chance) / (1.0 - chance)  # % of possible improvement over chance
        print(f"    {name:45s}: Acc = {acc:.3f}  (chance={chance:.2f}, lift={lift:.1%})")
        return acc

    # ── 5. Run all probes ───────────────────────────────────────────────────
    print("\n[10] Running probes on paired frames...")
    print("  ─── Raw z[t] (baseline, paired frames) ───")
    acc_raw_z = probe_classification(
        raw_t_embeddings, delta_motion_lbl,
        "Motion dir | raw z[t]", tr, te,
    )

    print("  ─── Δz = z[t+1] - z[t]  (DELTA PROBE) ───")
    acc_delta = probe_classification(
        delta_embeddings, delta_motion_lbl,
        "Motion dir | Δz (temporal delta)", tr, te,
    )

    print("  ─── [z_t ⊕ z_t+1] (upper bound, 2048-d) ───")
    acc_concat = probe_classification(
        concat_embeddings, delta_motion_lbl,
        "Motion dir | [z_t ⊕ z_t+1] concat", tr, te,
    )

    rand_delta   = rng.randn(*delta_embeddings.shape).astype(np.float32)
    print("  ─── Random baseline (Δz-shaped noise) ───")
    acc_rand = probe_classification(
        rand_delta, delta_motion_lbl,
        "Motion dir | random", tr, te,
    )

    # ── 6. Chart ──────────────────────────────────────────────────────────
    print("\n[11] Saving delta probe chart...")

    labels_chart  = [
        "Raw z[t]\n(single frame)",
        "Δz = z[t+1]−z[t]\n(delta probe)",
        "[z_t ⊕ z_{t+1}]\n(upper bound)",
        "Random\n(baseline)",
    ]
    scores_chart = [
        acc_raw_z   or 0,
        acc_delta   or 0,
        acc_concat  or 0,
        acc_rand    or 0,
    ]
    colors = ["#4fc3f7", "#66bb6a", "#ffa726", "#ef5350"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    bars = ax.bar(labels_chart, scores_chart, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
    for bar, score in zip(bars, scores_chart):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.012,
                f"{score:.3f}", ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")

    # Chance line (5 classes)
    n_classes = len(np.unique(delta_motion_lbl))
    chance    = 1.0 / n_classes
    ax.axhline(chance, color="#aaaaaa", linestyle="--", linewidth=1.5, label=f"Chance ({chance:.0%})")

    # Original full-dataset baseline (63.6%)
    ax.axhline(0.6363, color="#b39ddb", linestyle=":", linewidth=1.5, label="Original probe (full dataset, 63.6%)")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Motion Direction Accuracy", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=10)
    ax.set_title(
        "V-JEPA 2 — Temporal Delta Probe\n"
        f"Does Δz encode motion direction better than raw z?   (n={n_pairs} pairs)",
        color="white", fontsize=13, pad=14,
    )
    plt.tight_layout()
    plt.savefig("/output/delta_probe_summary.png", dpi=130, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── 7. Save results ───────────────────────────────────────────────────
    results = {
        "n_pairs":              n_pairs,
        "n_labeled_total":      n_labeled,
        "n_skipped_nonadjacent": skip_count,
        "motion_classes":       n_classes,
        "acc_raw_z_paired":     acc_raw_z,
        "acc_delta_z":          acc_delta,       # THE KEY NUMBER
        "acc_concat":           acc_concat,
        "acc_random":           acc_rand,
        "acc_original_fullset": 0.6363,          # from previous probe run
        "delta_improvement":    (acc_delta or 0) - (acc_raw_z or 0),
    }

    with open("/output/delta_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    output_vol.commit()

    print("\n=== DELTA PROBE RESULTS ===")
    print(f"  Raw z[t]      (paired):  {acc_raw_z:.3f}")
    print(f"  Δz temporal delta:       {acc_delta:.3f}   ← KEY RESULT")
    print(f"  [z_t ⊕ z_t+1] concat:   {acc_concat:.3f}")
    print(f"  Random baseline:         {acc_rand:.3f}")
    print(f"  Improvement (Δz vs raw): {(acc_delta or 0) - (acc_raw_z or 0):+.3f}")
    if acc_delta and acc_raw_z:
        if acc_delta > acc_raw_z + 0.05:
            print("\n✅ HYPOTHESIS CONFIRMED: Δz strongly encodes motion direction!")
            print("   → Latent dynamics model (z_t, a_t → z_t+1) is well-motivated.")
        elif acc_delta > acc_raw_z:
            print("\n🟡 PARTIAL: Δz improves motion accuracy but gain is small.")
            print("   → Consider: temporal window > 1, or motion is multi-token.")
        else:
            print("\n❌ HYPOTHESIS REJECTED: Δz does NOT improve motion prediction.")
            print("   → Motion may be encoded differently. Consider: optical flow features.")
    return results


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("Running V-JEPA 2 Temporal Delta Probe on Modal...")
    print("Note: If embeddings_cache.npz exists in volume, extraction is skipped.\n")

    results = run_delta_probe.remote(n_frames=10000, force_reextract=False)

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    for fname in ["delta_probe_results.json", "delta_probe_summary.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)],
                check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception as e:
            print(f"  Run manually: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")
    print("\nDone! Check decoder_output/delta_probe_summary.png")
