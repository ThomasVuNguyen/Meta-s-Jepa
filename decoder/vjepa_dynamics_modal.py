"""
V-JEPA 2 Latent Dynamics MLP — Phase 2 of Meta-s-Jepa
=======================================================
Trains a lightweight MLP to predict z_{t+1} from (z_t, a_t) in the
frozen V-JEPA 2 latent space, using DMControl environment rollouts.

Pipeline:
  1. Collect pixel rollouts from 3 DMControl envs (random policy)
       reacher-easy  (2-DOF arm,  action_dim=2)
       walker-walk   (bipedal,    action_dim=6)
       cheetah-run   (locomotion, action_dim=6)
  2. Extract V-JEPA 2 embeddings for every rendered frame (frozen)
  3. Train MLP: [z_t ⊕ a_t] → z_{t+1}  (MSE loss, Adam)
  4. Validate: probe predicted z_{t+1} for spatial accuracy
       - Train YOLO on env frames for ground-truth labels
       - Run linear probe on predicted embeddings: XY R², size R²
       - Compare with probe on true z_{t+1} (upper bound)
  5. Save model + results

Outputs (saved to vjepa2-decoder-output volume):
  dynamics_mlp.pt              — trained model weights
  dynamics_train_loss.png      — training loss curve
  dynamics_validation.json     — probe accuracy on predicted vs true z
  dynamics_summary.png         — bar chart: predicted vs true probe scores

Why MSE in latent space?
  Validated by CURLing the Dream (Kich et al. 2024):
  L = ||MLP(z_t, a_t) - z_{t+1}||²  achieves SOTA on DMC without reconstruction.
  Predicted z can then be probed for spatial validity.
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-dynamics")

model_cache  = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol   = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol  = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        # MuJoCo system deps
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf",
        "xvfb", "ffmpeg",
    )
    .run_commands(
        "/opt/conda/bin/pip install opencv-python-headless",
        "/opt/conda/bin/pip install --no-deps ultralytics",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scikit-learn scipy tqdm",
        # DMControl + MuJoCo
        "/opt/conda/bin/pip install dm_control mujoco",
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Collect DMControl rollouts
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/rollouts": rollout_vol},
)
def collect_rollouts(
    envs: list = None,
    n_episodes: int = 20,
    episode_len: int = 200,
    render_size: int = 256,
    force_recollect: bool = False,
):
    """
    Collect pixel rollouts from DMControl environments.
    Saves (frames, actions) per env to the rollout volume.

    n_episodes * episode_len = total transitions per env
    Default: 20 * 200 = 4,000 transitions per env, 12,000 total
    """
    import os, json, numpy as np
    from pathlib import Path

    if envs is None:
        envs = [
            ("reacher", "easy"),
            ("walker",  "walk"),
            ("cheetah", "run"),
        ]

    # Headless rendering via OSMesa
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    from dm_control import suite

    results = {}
    for domain, task in envs:
        env_key = f"{domain}_{task}"
        save_dir = Path(f"/rollouts/{env_key}")
        frames_path  = save_dir / "frames.npy"
        actions_path = save_dir / "actions.npy"
        meta_path    = save_dir / "meta.json"

        if frames_path.exists() and not force_recollect:
            meta = json.loads(meta_path.read_text())
            print(f"[CACHE] {env_key}: {meta['n_transitions']} transitions already collected")
            results[env_key] = meta
            continue

        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[COLLECT] {env_key} — {n_episodes} episodes × {episode_len} steps")

        env = suite.load(domain_name=domain, task_name=task,
                         task_kwargs={"random": 42})
        action_spec = env.action_spec()
        action_dim  = action_spec.shape[0]
        print(f"  action_dim={action_dim}  action range=[{action_spec.minimum[0]:.2f}, {action_spec.maximum[0]:.2f}]")

        rng = np.random.RandomState(0)
        all_frames  = []  # (N, H, W, 3) uint8
        all_actions = []  # (N, action_dim)

        for ep in range(n_episodes):
            ts = env.reset()
            ep_frames  = []
            ep_actions = []
            for step in range(episode_len):
                # Render pixel frame
                frame = env.physics.render(
                    height=render_size, width=render_size, camera_id=0
                )
                # Random action (uniform in valid range)
                action = rng.uniform(
                    action_spec.minimum, action_spec.maximum
                ).astype(np.float32)
                ts = env.step(action)

                ep_frames.append(frame)
                ep_actions.append(action)

                if ts.last():
                    break

            # Append consecutive (t, t+1) pairs — skip last frame per episode
            for t in range(len(ep_frames) - 1):
                all_frames.append(ep_frames[t])
                all_actions.append(ep_actions[t])

            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}/{n_episodes} done")

        frames_arr  = np.array(all_frames,  dtype=np.uint8)    # (N, H, W, 3)
        actions_arr = np.array(all_actions, dtype=np.float32)  # (N, action_dim)
        n_transitions = len(frames_arr)
        print(f"  → {n_transitions} transitions collected")
        print(f"  → frames: {frames_arr.shape}, actions: {actions_arr.shape}")

        np.save(str(frames_path),  frames_arr)
        np.save(str(actions_path), actions_arr)
        meta = {
            "env_key": env_key,
            "domain": domain,
            "task": task,
            "n_transitions": n_transitions,
            "action_dim": action_dim,
            "render_size": render_size,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        rollout_vol.commit()
        results[env_key] = meta
        print(f"  ✓ Saved to /rollouts/{env_key}/")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 + 3: Extract embeddings & Train dynamics MLP
# ─────────────────────────────────────────────────────────────────────────────

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
def train_dynamics(
    envs: list = None,
    hidden_dim: int = 512,
    n_layers: int = 3,
    lr: float = 1e-3,
    n_epochs: int = 50,
    batch_size: int = 256,
):
    """
    Extract V-JEPA 2 embeddings, train MLP dynamics, validate.
    """
    import os, json, numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    DEVICE = "cuda"

    if envs is None:
        envs = ["reacher_easy", "walker_walk", "cheetah_run"]

    # ── Load V-JEPA 2 ────────────────────────────────────────────────────────
    print("[1] Loading V-JEPA 2...")
    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        trust_remote_code=True,
        cache_dir="/cache/hf",
    ).to(DEVICE, dtype=torch.float16).eval()
    print(f"    {sum(p.numel() for p in vjepa.parameters()):,} params (frozen)")

    ET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def extract_embeddings(frames_np, batch_size=32):
        """frames_np: (N, H, W, 3) uint8 → returns (N, 1024) float32"""
        embs = []
        for start in range(0, len(frames_np), batch_size):
            batch = frames_np[start:start + batch_size]
            clips = []
            for frame in batch:
                img = Image.fromarray(frame)
                t = ET(img)
                clips.append(t.unsqueeze(0).repeat(8, 1, 1, 1))  # [8,C,H,W]
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips)
                emb = out.last_hidden_state.mean(dim=1).cpu().float()
            embs.append(emb)
            if (start // batch_size) % 10 == 0:
                print(f"    {start}/{len(frames_np)} frames encoded")
        return torch.cat(embs, dim=0).numpy()

    # ── Collect embeddings across all envs ───────────────────────────────────
    print("\n[2] Extracting embeddings from rollouts...")
    all_z_t    = []
    all_z_t1   = []
    all_actions = []
    env_labels  = []  # for per-env analysis

    for env_key in envs:
        rd = Path(f"/rollouts/{env_key}")
        if not rd.exists():
            print(f"  ⚠ {env_key}: no rollouts found. Run collect_rollouts first.")
            continue

        frames_path  = rd / "frames.npy"
        actions_path = rd / "actions.npy"
        meta         = json.loads((rd / "meta.json").read_text())

        print(f"\n  [{env_key}] Loading {meta['n_transitions']} transitions...")
        frames_t  = np.load(str(frames_path))   # (N, H, W, 3)
        actions_t = np.load(str(actions_path))  # (N, action_dim)

        # frames_t[i] = frame at time t, so frame t+1 is frames_t[i+1]
        # (we stored consecutive frames, so frame_t and frame_{t+1} are adjacent)
        # need to extract BOTH z_t and z_{t+1}
        print(f"  Extracting z_t embeddings...")
        z_t  = extract_embeddings(frames_t)       # (N, 1024)
        print(f"  Extracting z_{{t+1}} embeddings (shifted by 1)...")
        # Shift: z_{t+1} = embedding of the *next* frame
        # frames_t[i] is paired with action actions_t[i], next frame is frames_t[i+1]
        # So we extract embeddings for frames_t[1:] as z_{t+1}
        z_t1 = extract_embeddings(frames_t[1:])  # (N-1, 1024)

        # Align: drop last z_t and action (no paired z_{t+1} for last frame)
        z_t_aligned      = z_t[:-1]           # (N-1, 1024)
        actions_aligned  = actions_t[:-1]     # (N-1, action_dim)

        print(f"  {env_key}: {len(z_t_aligned)} (z_t, a_t, z_{{t+1}}) pairs")

        all_z_t.append(z_t_aligned)
        all_z_t1.append(z_t1)
        all_actions.append(actions_aligned)
        env_labels.extend([env_key] * len(z_t_aligned))

    if not all_z_t:
        raise RuntimeError("No rollout data found. Run collect_rollouts first.")

    # Pad actions to max_action_dim (different envs have different action dims)
    max_action_dim = max(a.shape[1] for a in all_actions)
    padded_actions = []
    for a in all_actions:
        pad = np.zeros((a.shape[0], max_action_dim - a.shape[1]), dtype=np.float32)
        padded_actions.append(np.concatenate([a, pad], axis=1))

    Z_t   = np.concatenate(all_z_t,       axis=0).astype(np.float32)  # (M, 1024)
    Z_t1  = np.concatenate(all_z_t1,      axis=0).astype(np.float32)  # (M, 1024)
    A_t   = np.concatenate(padded_actions, axis=0).astype(np.float32) # (M, max_action_dim)

    N = len(Z_t)
    print(f"\n  Total: {N} (z_t, a_t, z_{{t+1}}) tuples across {len(envs)} envs")
    print(f"  z_t shape: {Z_t.shape}, actions shape: {A_t.shape}")

    # ── MLP Architecture ─────────────────────────────────────────────────────
    print(f"\n[3] Building dynamics MLP...")
    z_dim    = Z_t.shape[1]      # 1024
    a_dim    = A_t.shape[1]      # max_action_dim
    input_dim = z_dim + a_dim    # 1024 + action_dim

    class DynamicsMLP(nn.Module):
        """
        Lightweight MLP: [z_t ⊕ a_t] → z_{t+1}
        Architecture from CURLing the Dream (Kich et al. 2024)
        """
        def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
            super().__init__()
            layers = []
            in_d = input_dim
            for i in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, output_dim))  # no activation on output
            self.net = nn.Sequential(*layers)

        def forward(self, z, a):
            x = torch.cat([z, a], dim=-1)
            return self.net(x)

    model = DynamicsMLP(input_dim, hidden_dim, z_dim, n_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DynamicsMLP: {n_params:,} params  ({n_layers} layers, hidden={hidden_dim})")
    print(f"  Input: {input_dim}-d  →  Output: {z_dim}-d")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[4] Training dynamics MLP ({n_epochs} epochs, batch={batch_size}, lr={lr})...")

    rng   = np.random.RandomState(42)
    perm  = rng.permutation(N)
    split = int(0.85 * N)
    tr_idx, te_idx = perm[:split], perm[split:]

    Z_t_tr  = torch.tensor(Z_t[tr_idx]).to(DEVICE)
    Z_t1_tr = torch.tensor(Z_t1[tr_idx]).to(DEVICE)
    A_t_tr  = torch.tensor(A_t[tr_idx]).to(DEVICE)

    Z_t_te  = torch.tensor(Z_t[te_idx]).to(DEVICE)
    Z_t1_te = torch.tensor(Z_t1[te_idx]).to(DEVICE)
    A_t_te  = torch.tensor(A_t[te_idx]).to(DEVICE)

    dataset    = TensorDataset(Z_t_tr, A_t_tr, Z_t1_tr)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion  = nn.MSELoss()

    train_losses = []
    val_losses   = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for z_b, a_b, z1_b in loader:
            optimizer.zero_grad()
            z_pred = model(z_b, a_b)
            loss   = criterion(z_pred, z1_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(z_b)
        scheduler.step()
        train_losses.append(epoch_loss / len(tr_idx))

        # Validation
        model.eval()
        with torch.no_grad():
            z_pred_val = model(Z_t_te, A_t_te)
            val_loss   = criterion(z_pred_val, Z_t1_te).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}")

    print(f"\n  Final val_loss: {val_losses[-1]:.4f}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#1a1a1a")
    ax.plot(train_losses, color="#4fc3f7", label="Train MSE", linewidth=1.5)
    ax.plot(val_losses,   color="#ff8a65", label="Val MSE",   linewidth=1.5)
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("MSE Loss", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.legend(facecolor="#222", labelcolor="white")
    ax.set_title(
        f"Dynamics MLP Training  ({', '.join(envs)})\n"
        f"[z_t ⊕ a_t] → z_{{t+1}},  {n_params:,} params",
        color="white"
    )
    plt.tight_layout()
    plt.savefig("/output/dynamics_train_loss.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    # ── Validation: probe predicted z_{t+1} for spatial accuracy ─────────────
    print("\n[5] Validating: probing predicted z_{t+1} for spatial information...")
    print("    Generating predicted embeddings for test set...")

    model.eval()
    with torch.no_grad():
        Z_pred_te = model(Z_t_te, A_t_te).cpu().numpy()   # predicted z_{t+1}
    Z_true_te = Z_t1[te_idx]                               # true z_{t+1}
    Z_t_np    = Z_t[te_idx]                                # z_t context

    print("    Running YOLO on test frames to get spatial labels...")
    # Get the original rendered frames for test set to generate YOLO labels
    # We'll use z_t frames (already rendered), run YOLO to get XY/size labels
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")

    # Collect test frames across envs for YOLO labeling
    # Map test indices back to frames
    test_env_frames = []
    test_env_labels_xy  = []
    test_env_labels_wh  = []

    offset = 0
    for i, env_key in enumerate(envs):
        rd = Path(f"/rollouts/{env_key}")
        if not rd.exists():
            continue
        frames_t = np.load(str(rd / "frames.npy"))
        n_env    = len(frames_t) - 1  # number of pairs from this env
        # find which test indices fall in this env's range
        env_test_mask = [
            idx for idx in range(len(te_idx))
            if offset <= te_idx[idx] < offset + n_env
        ]
        for local_i in env_test_mask:
            global_frame_idx = te_idx[local_i] - offset
            frame = Image.fromarray(frames_t[global_frame_idx])
            test_env_frames.append((local_i, frame))
        offset += n_env

    # Run YOLO on collected test frames
    labeled_map = {}  # local_test_idx → (cx, cy, bw, bh)
    BATCH = 32
    frame_items = test_env_frames
    for b_start in range(0, len(frame_items), BATCH):
        batch = frame_items[b_start:b_start + BATCH]
        imgs  = [f for _, f in batch]
        idxs  = [i for i, _ in batch]
        res_list = yolo_model(imgs, verbose=False, conf=0.25, device=0)
        W = H = 256
        for idx, res in zip(idxs, res_list):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best  = int(boxes.conf.argmax())
            xyxy  = boxes.xyxy[best].cpu().numpy()
            cx = float((xyxy[0] + xyxy[2]) / 2 / W)
            cy = float((xyxy[1] + xyxy[3]) / 2 / H)
            bw = float((xyxy[2] - xyxy[0]) / W)
            bh = float((xyxy[3] - xyxy[1]) / H)
            labeled_map[idx] = (cx, cy, bw, bh)

    if len(labeled_map) < 50:
        print(f"  ⚠ Only {len(labeled_map)} YOLO detections in test set — robots are harder to detect than natural videos")
        print("    Falling back to cosine similarity as validation metric.")
        # Fallback: cosine similarity between predicted and true z_{t+1}
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = float(np.mean(np.diag(cosine_similarity(Z_pred_te, Z_true_te))))
        cos_sim_same = float(np.mean(np.diag(cosine_similarity(Z_t_np, Z_true_te))))
        print(f"  cos_sim(z_pred, z_true): {cos_sim:.4f}")
        print(f"  cos_sim(z_t,    z_true): {cos_sim_same:.4f}  (copying z_t baseline)")
        val_results = {
            "method": "cosine_similarity_fallback",
            "cos_sim_predicted_vs_true": cos_sim,
            "cos_sim_zt_vs_ztplus1":     cos_sim_same,
            "n_yolo_detections": len(labeled_map),
        }
    else:
        # Full spatial probe validation
        labeled_idxs = sorted(labeled_map.keys())
        labels_xy = np.array([[labeled_map[i][0], labeled_map[i][1]] for i in labeled_idxs], dtype=np.float32)
        labels_wh = np.array([[labeled_map[i][2], labeled_map[i][3]] for i in labeled_idxs], dtype=np.float32)

        Z_pred_labeled = Z_pred_te[labeled_idxs]
        Z_true_labeled = Z_true_te[labeled_idxs]

        def probe_r2(feat, lab, name):
            scaler = StandardScaler()
            X = scaler.fit_transform(feat)
            ridge = Ridge(alpha=1.0)
            split = int(0.8 * len(X))
            ridge.fit(X[:split], lab[:split])
            pred = ridge.predict(X[split:])
            r2 = float(r2_score(lab[split:], pred))
            print(f"    {name:40s}: R² = {r2:.3f}")
            return r2

        print(f"\n  Labeled {len(labeled_idxs)} test frames via YOLO")
        print("  --- True z_{t+1} (upper bound) ---")
        r2_xy_true = probe_r2(Z_true_labeled, labels_xy, "XY position  | true z_{t+1}")
        r2_wh_true = probe_r2(Z_true_labeled, labels_wh, "Object size  | true z_{t+1}")
        print("  --- Predicted z_{t+1} (dynamics model) ---")
        r2_xy_pred = probe_r2(Z_pred_labeled, labels_xy, "XY position  | predicted z_{t+1}")
        r2_wh_pred = probe_r2(Z_pred_labeled, labels_wh, "Object size  | predicted z_{t+1}")
        print("  --- z_t (copying baseline) ---")
        Z_zt_labeled = Z_t_np[labeled_idxs]
        r2_xy_copy = probe_r2(Z_zt_labeled, labels_xy, "XY position  | z_t (copy baseline)")
        r2_wh_copy = probe_r2(Z_zt_labeled, labels_wh, "Object size  | z_t (copy baseline)")

        val_results = {
            "method": "spatial_probe",
            "n_labeled": len(labeled_idxs),
            "r2_xy_true_z":      r2_xy_true,
            "r2_wh_true_z":      r2_wh_true,
            "r2_xy_predicted_z": r2_xy_pred,
            "r2_wh_predicted_z": r2_wh_pred,
            "r2_xy_copy_zt":     r2_xy_copy,
            "r2_wh_copy_zt":     r2_wh_copy,
            "xy_retention_pct":  100 * r2_xy_pred / max(r2_xy_true, 1e-6),
            "wh_retention_pct":  100 * r2_wh_pred / max(r2_wh_true, 1e-6),
        }

        # Chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#111")
        for ax in axes: ax.set_facecolor("#1a1a1a")

        for ax, (metric, true_v, pred_v, copy_v) in zip(axes, [
            ("XY Position R²", r2_xy_true, r2_xy_pred, r2_xy_copy),
            ("Object Size R²", r2_wh_true, r2_wh_pred, r2_wh_copy),
        ]):
            bars = ax.bar(
                ["True z_{t+1}\n(upper bound)", "Predicted z_{t+1}\n(dynamics)", "z_t copy\n(baseline)"],
                [true_v, pred_v, copy_v],
                color=["#ffa726", "#66bb6a", "#ef5350"], width=0.55,
                edgecolor="white", linewidth=0.5
            )
            for bar, val in zip(bars, [true_v, pred_v, copy_v]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.1)
            ax.set_title(metric, color="white", fontsize=12)
            ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

        fig.suptitle("Dynamics MLP Validation — Does predicted z_{t+1} preserve spatial information?",
                     color="white", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig("/output/dynamics_summary.png", dpi=120, bbox_inches="tight", facecolor="#111")
        plt.close()

    # ── Save model + results ──────────────────────────────────────────────────
    print("\n[6] Saving model and results...")
    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   input_dim,
        "hidden_dim":  hidden_dim,
        "z_dim":       z_dim,
        "a_dim":       a_dim,
        "n_layers":    n_layers,
        "max_action_dim": max_action_dim,
        "envs":        envs,
        "n_params":    n_params,
        "final_val_loss": val_losses[-1],
    }, "/output/dynamics_mlp.pt")

    results = {
        "n_transitions_total": N,
        "envs": envs,
        "n_params": n_params,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "final_train_loss": train_losses[-1],
        "final_val_loss":   val_losses[-1],
        **val_results,
    }
    import json
    with open("/output/dynamics_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    output_vol.commit()

    print("\n=== DYNAMICS MLP RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    ENVS = [
        ("reacher", "easy"),
        ("walker",  "walk"),
        ("cheetah", "run"),
    ]
    ENV_KEYS = [f"{d}_{t}" for d, t in ENVS]

    print("=" * 60)
    print("META-S-JEPA  Phase 2: Latent Dynamics MLP")
    print("=" * 60)

    # Stage 1: Collect rollouts (CPU, no GPU needed)
    print("\n[Stage 1] Collecting DMControl rollouts...")
    rollout_meta = collect_rollouts.remote(
        envs=ENVS,
        n_episodes=25,
        episode_len=200,
        render_size=256,
        force_recollect=False,
    )
    print("\nRollout summary:")
    for env_key, meta in rollout_meta.items():
        print(f"  {env_key}: {meta['n_transitions']} transitions, action_dim={meta['action_dim']}")

    # Stage 2+3: Train on A10G
    print("\n[Stage 2+3] Extracting embeddings + training dynamics MLP on A10G...")
    results = train_dynamics.remote(
        envs=ENV_KEYS,
        hidden_dim=512,
        n_layers=3,
        lr=1e-3,
        n_epochs=50,
        batch_size=256,
    )

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Download outputs
    out = Path("./decoder_output")
    out.mkdir(exist_ok=True)
    for fname in ["dynamics_validation.json", "dynamics_train_loss.png", "dynamics_summary.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fname, str(out / fname)],
                check=True,
            )
            print(f"  ✓ Downloaded {fname}")
        except Exception as e:
            print(f"  Run manually: modal volume get vjepa2-decoder-output {fname} decoder_output/{fname}")

    print("\nDone! Check decoder_output/dynamics_summary.png")
