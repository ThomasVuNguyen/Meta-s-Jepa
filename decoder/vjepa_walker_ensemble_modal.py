"""
V-JEPA 2 — Experiment 6: Walker-Walk Transfer + Ensemble Uncertainty
====================================================================
Two questions in one run:

Part A — Environment Scaling
  Does the Phase-4-style pipeline (embed → learn dynamics → MPC) work
  on a harder, higher-DOF body?  We use `walker-walk` (9-DOF humanoid
  torso vs 2-DOF reacher arm).  Zero-shot transfer of the V-JEPA encoder
  (no additional pre-training), collect data and train a new dynamics MLP.

Part B — Dynamics Ensemble Uncertainty
  Train 5 independent MLPs on the same data.  Use disagreement between
  ensemble members as a proxy for model uncertainty.  Feed uncertainty
  into the MPC cost (penalise high-variance rollouts) to see if it
  avoids out-of-distribution planning.

Hypothesis A:  V-JEPA's general visual encoder should generalise to walker
  even without fine-tuning; the MPC policy should beat random with a
  domain-adapted dynamics MLP.

Hypothesis B:  Ensemble disagreement will be higher at the start of a
  plan (compounding error) and penalising it will cause CEM to prefer
  shorter effective horizons naturally, reducing the horizon-tuning burden.

Compute estimate: ~120 min A10G, ~$2.20
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-walker-ensemble")

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
    timeout=14400,
    volumes={
        "/cache":    model_cache,
        "/output":   output_vol,
        "/rollouts": rollout_vol,
    },
)
def walker_ensemble_experiment(
    # Data collection
    walker_episodes:      int = 200,
    walker_ep_steps:      int = 100,
    # Dynamics MLP
    z_dim:                int = 1024,
    a_pad_dim:            int = 6,
    hidden:               int = 512,
    n_layers:             int = 3,
    train_epochs:         int = 60,
    batch_size:           int = 256,
    lr:                   float = 2e-4,
    # Ensemble
    n_ensemble:           int = 5,
    # MPC
    horizon:              int = 30,   # shorter for walker (harder env)
    n_candidates:         int = 256,
    n_elites:             int = 32,
    n_cem_iters:          int = 5,
    mpc_steps:            int = 100,  # walker episodes are longer
    # Evaluation
    n_eval_episodes:      int = 10,
    uncertainty_penalty:  float = 0.5,  # weight for ensemble disagreement
):
    import os, json, time
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    DEVICE = "cuda"

    print("[1] Loading V-JEPA 2...")
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

    def embed_single(f):
        img  = Image.fromarray(f)
        clip = ET(img).unsqueeze(0).repeat(8,1,1,1).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            return vjepa(pixel_values_videos=clip).last_hidden_state.mean(dim=1).squeeze(0).float()

    def embed_batch(frames_np, bs=32):
        embs = []
        for s in range(0, len(frames_np), bs):
            batch = frames_np[s:s+bs]
            clips = []
            for f in batch:
                img  = Image.fromarray(f)
                clip = ET(img).unsqueeze(0).repeat(8,1,1,1)
                clips.append(clip)
            clips = torch.stack(clips).to(DEVICE, dtype=torch.float16)
            with torch.no_grad():
                out = vjepa(pixel_values_videos=clips)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(embs, dim=0)

    def pad_action(a, pad_to=6):
        return np.concatenate([a, np.zeros(max(0, pad_to - len(a)), dtype=np.float32)])

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

    # ─── Part A: collect walker-walk data ─────────────────────────────────
    print(f"\n[2] Collecting walker-walk data ({walker_episodes} episodes × {walker_ep_steps} steps)...")
    from dm_control import suite

    rng = np.random.RandomState(22)
    all_frames, all_actions = [], []
    for ep in range(walker_episodes):
        env   = suite.load("walker", "walk",  task_kwargs={"random": rng.randint(0,10000)})
        aspec = env.action_spec()
        env.reset()
        frames_ep, actions_ep = [], []
        for _ in range(walker_ep_steps):
            frames_ep.append(env.physics.render(height=256, width=256, camera_id=0))
            action = rng.uniform(aspec.minimum, aspec.maximum)
            actions_ep.append(pad_action(action))
            env.step(action)
        all_frames.extend(frames_ep)
        all_actions.extend(actions_ep)
        if (ep + 1) % 50 == 0:
            print(f"    {ep+1}/{walker_episodes} episodes ({len(all_frames)} frames)")

    frames_np  = np.array(all_frames,  dtype=np.uint8)
    actions_np = np.array(all_actions, dtype=np.float32)
    print(f"    Total: {len(frames_np)} frames")

    # Save to rollout volume for future use
    walker_dir = Path("/rollouts/walker_walk")
    walker_dir.mkdir(exist_ok=True)
    np.save(str(walker_dir / "frames.npy"),  frames_np)
    np.save(str(walker_dir / "actions.npy"), actions_np)
    rollout_vol.commit()

    print(f"[3] Embedding {len(frames_np)} walker frames...")
    t0 = time.time()
    Z_all = embed_batch(frames_np)
    print(f"    Embedded in {(time.time()-t0)/60:.1f} min")

    Z_t  = Z_all[:-1].numpy()
    A_t  = actions_np[:-1]
    Z_t1 = Z_all[1:].numpy()
    N    = len(Z_t)
    print(f"    Dataset: {N} transitions")

    perm  = np.random.RandomState(0).permutation(N)
    split = int(0.85 * N)
    tr, te = perm[:split], perm[split:]

    Z_tr  = torch.tensor(Z_t[tr]);   A_tr  = torch.tensor(A_t[tr]);   Z1_tr = torch.tensor(Z_t1[tr])
    Z_te  = torch.tensor(Z_t[te]);   A_te  = torch.tensor(A_t[te]);   Z1_te = torch.tensor(Z_t1[te])
    criterion = nn.MSELoss()

    # ─── Part A: single best-model training ───────────────────────────────
    print(f"\n[4] Training single dynamics MLP for walker-walk ({train_epochs} epochs)...")
    main_model = DynamicsMLP().to(DEVICE)
    opt = torch.optim.AdamW(main_model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_epochs)
    ds  = TensorDataset(Z_tr, A_tr, Z1_tr)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    train_losses_main, val_losses_main = [], []
    for epoch in range(1, train_epochs + 1):
        main_model.train()
        ep_loss = 0.0
        for z_b, a_b, z1_b in dl:
            z_b  = z_b.to(DEVICE); a_b = a_b.to(DEVICE); z1_b = z1_b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(main_model(z_b, a_b), z1_b)
            loss.backward(); nn.utils.clip_grad_norm_(main_model.parameters(), 1.0); opt.step()
            ep_loss += loss.item() * len(z_b)
        sch.step()
        main_model.eval()
        with torch.no_grad():
            vl = criterion(main_model(Z_te.to(DEVICE), A_te.to(DEVICE)), Z1_te.to(DEVICE)).item()
        train_losses_main.append(ep_loss / len(ds))
        val_losses_main.append(vl)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{train_epochs}  val={vl:.4f}")

    torch.save({
        "model_state": main_model.state_dict(), "env": "walker-walk",
        "final_val_loss": val_losses_main[-1],
    }, "/output/walker_dynamics_mlp.pt")

    # ─── Part B: ensemble training ────────────────────────────────────────
    print(f"\n[5] Training ensemble of {n_ensemble} MLPs...")
    ensemble = []
    ensemble_val_losses = []
    for m_idx in range(n_ensemble):
        model = DynamicsMLP().to(DEVICE)
        # Each member sees a different bootstrap sample
        boot_idx = np.random.RandomState(m_idx * 17).choice(len(tr), len(tr), replace=True)
        ds_m = TensorDataset(Z_tr[boot_idx], A_tr[boot_idx], Z1_tr[boot_idx])
        dl_m = DataLoader(ds_m, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        opt_m = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sch_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=train_epochs)
        for epoch in range(train_epochs):
            model.train()
            for z_b, a_b, z1_b in dl_m:
                z_b  = z_b.to(DEVICE); a_b = a_b.to(DEVICE); z1_b = z1_b.to(DEVICE)
                opt_m.zero_grad()
                criterion(model(z_b, a_b), z1_b).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_m.step()
            sch_m.step()
        model.eval()
        with torch.no_grad():
            vl_m = criterion(model(Z_te.to(DEVICE), A_te.to(DEVICE)), Z1_te.to(DEVICE)).item()
        ensemble_val_losses.append(vl_m)
        ensemble.append(model)
        print(f"    member {m_idx+1}/{n_ensemble}  val={vl_m:.4f}")

    def ensemble_predict_with_uncertainty(z_batch, a_batch):
        """Returns mean prediction and std (uncertainty) across ensemble."""
        preds = []
        for m in ensemble:
            with torch.no_grad():
                preds.append(m(z_batch, a_batch))
        preds = torch.stack(preds, dim=0)  # (M, N, z_dim)
        return preds.mean(dim=0), preds.std(dim=0).mean(dim=-1)  # mean pred, scalar uncertainty per sample

    # ─── Evaluation: single MLP vs ensemble (with + without penalty) ──────
    print(f"\n[6] Evaluating on walker-walk ({n_eval_episodes} episodes)...")
    results_single, results_ens_no_penalty, results_ens_penalty = [], [], []
    eval_rng = np.random.RandomState(99)

    for ep in range(n_eval_episodes):
        # ---- helper: run one walker episode with given dynamics function ----
        def run_episode(dynamics_fn, uncertainty_weight=0.0):
            env = suite.load("walker", "walk", task_kwargs={"random": eval_rng.randint(0, 10000)})
            aspec = env.action_spec()
            env.reset()
            total_reward = 0.0
            for _ in range(mpc_steps):
                z_curr = embed_single(env.physics.render(height=256, width=256, camera_id=0)).to(DEVICE)
                mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
                sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
                for _ in range(n_cem_iters):
                    eps      = eval_rng.randn(n_candidates, horizon, a_pad_dim).astype(np.float32)
                    act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
                    act_t    = torch.tensor(act_seqs, device=DEVICE)
                    z_c      = z_curr.unsqueeze(0).expand(n_candidates, -1).clone()
                    total_unc = torch.zeros(n_candidates, device=DEVICE)
                    with torch.no_grad():
                        for t in range(horizon):
                            result = dynamics_fn(z_c, act_t[:, t, :])
                            if isinstance(result, tuple):
                                z_c, unc = result
                                total_unc += unc * (uncertainty_weight > 0)
                            else:
                                z_c = result
                    # Cost: L2 distance to mean origin (walker: maximise speed → penalise staying still)
                    # Proxy: minimise distance from target coordinate (centre-of-mass forward)
                    # For walker-walk we use the z_c norm as a proxy reward
                    costs = -z_c.norm(dim=-1) + uncertainty_weight * total_unc
                    costs = costs.cpu().numpy()
                    elite_idx = np.argsort(costs)[:n_elites]
                    mu  = act_seqs[elite_idx].mean(axis=0)
                    sig = act_seqs[elite_idx].std(axis=0) + 1e-6
                timestep = env.step(np.clip(mu[0, :len(aspec.minimum)], aspec.minimum, aspec.maximum))
                total_reward += float(timestep.reward or 0.0)
            # Random baseline
            env2 = suite.load("walker", "walk", task_kwargs={"random": eval_rng.randint(0, 10000)})
            aspec2 = env2.action_spec(); env2.reset()
            rand_reward = 0.0
            for _ in range(mpc_steps):
                ts = env2.step(eval_rng.uniform(aspec2.minimum, aspec2.maximum))
                rand_reward += float(ts.reward or 0.0)
            return total_reward, rand_reward

        # Single MLP
        def single_fn(z, a):
            with torch.no_grad(): return main_model(z, a)
        r_single, r_rand = run_episode(single_fn, uncertainty_weight=0.0)
        results_single.append({"mpc_reward": r_single, "rand_reward": r_rand,
                                "win": r_single > r_rand})

        # Ensemble without penalty
        def ens_fn(z, a):
            mean_pred, unc = ensemble_predict_with_uncertainty(z, a)
            return mean_pred, unc
        r_ens, r_rand2 = run_episode(ens_fn, uncertainty_weight=0.0)
        results_ens_no_penalty.append({"mpc_reward": r_ens, "rand_reward": r_rand2,
                                        "win": r_ens > r_rand2})

        # Ensemble with uncertainty penalty
        r_ens_p, r_rand3 = run_episode(ens_fn, uncertainty_weight=uncertainty_penalty)
        results_ens_penalty.append({"mpc_reward": r_ens_p, "rand_reward": r_rand3,
                                     "win": r_ens_p > r_rand3})

        print(f"  ep{ep+1}:  single={r_single:.1f}  ens={r_ens:.1f}  ens+unc={r_ens_p:.1f}  rand≈{r_rand:.1f}")

    def summary(res):
        avg_mpc  = float(np.mean([r["mpc_reward"]  for r in res]))
        avg_rand = float(np.mean([r["rand_reward"] for r in res]))
        win_pct  = float(np.mean([r["win"]         for r in res])) * 100
        return {"avg_mpc_reward": avg_mpc, "avg_rand_reward": avg_rand, "win_pct": win_pct}

    s_single  = summary(results_single)
    s_ens     = summary(results_ens_no_penalty)
    s_ens_pen = summary(results_ens_penalty)

    print(f"\n  Single MLP:      win={s_single['win_pct']:.0f}%  MPC={s_single['avg_mpc_reward']:.1f}  rand={s_single['avg_rand_reward']:.1f}")
    print(f"  Ensemble:        win={s_ens['win_pct']:.0f}%  MPC={s_ens['avg_mpc_reward']:.1f}")
    print(f"  Ens+uncertainty: win={s_ens_pen['win_pct']:.0f}%  MPC={s_ens_pen['avg_mpc_reward']:.1f}")

    # ─── Charts ────────────────────────────────────────────────────────────
    labels = ["Single MLP", "Ensemble", "Ensemble\n+ Unc. Penalty"]
    wins   = [s_single["win_pct"], s_ens["win_pct"], s_ens_pen["win_pct"]]
    mpcs   = [s_single["avg_mpc_reward"], s_ens["avg_mpc_reward"], s_ens_pen["avg_mpc_reward"]]
    rands  = [s_single["avg_rand_reward"], s_ens["avg_rand_reward"], s_ens_pen["avg_rand_reward"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    for ax in [ax1, ax2]: ax.set_facecolor("#1a1a1a")

    x = np.arange(len(labels))
    ax1.bar(x, mpcs,  width=0.35, label="MPC",    color="#4fc3f7", edgecolor="white")
    ax1.bar(x+0.35, rands, width=0.35, label="Random", color="#ef5350", edgecolor="white")
    ax1.set_xticks(x + 0.175); ax1.set_xticklabels(labels, color="white", fontsize=9)
    ax1.set_ylabel("Avg cumulative reward", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white")
    ax1.set_title("Walker-Walk: Avg Cumulative Reward", color="white")

    colors = ["#ef9a9a" if w < 50 else "#66bb6a" for w in wins]
    ax2.bar(x + 0.175, wins, width=0.5, color=colors, edgecolor="white")
    ax2.axhline(50, color="#ffa726", ls="--", lw=1.5, label="Random (50%)")
    for xi, w in zip(x, wins):
        ax2.text(xi + 0.175, w + 1.5, f"{w:.0f}%", ha="center", color="white", fontsize=11, fontweight="bold")
    ax2.set_xticks(x + 0.175); ax2.set_xticklabels(labels, color="white", fontsize=9)
    ax2.set_ylabel("Win rate (%)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white")
    ax2.set_title("Walker-Walk: Win Rate vs Random", color="white")

    fig.suptitle(
        "Experiment 6: Walker-Walk Transfer + Ensemble Uncertainty\n"
        f"V-JEPA encoder (frozen) | {walker_episodes} random episodes | horizon T={horizon}",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("/output/walker_ensemble_results.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    final = {
        "experiment": 6,
        "env": "walker-walk",
        "n_episodes_train": walker_episodes,
        "n_transitions": N,
        "n_ensemble": n_ensemble,
        "horizon": horizon,
        "single_mlp": {**s_single, "final_val_loss": val_losses_main[-1]},
        "ensemble": {**s_ens, "avg_val_loss": float(np.mean(ensemble_val_losses))},
        "ensemble_with_uncertainty": s_ens_pen,
    }
    with open("/output/walker_ensemble_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()

    print("\n=== EXPERIMENT 6 COMPLETE ===")
    print(f"  Single MLP:   win={s_single['win_pct']:.0f}%  val={val_losses_main[-1]:.4f}")
    print(f"  Ensemble:     win={s_ens['win_pct']:.0f}%")
    print(f"  Ens+unc:      win={s_ens_pen['win_pct']:.0f}%")
    return final


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Experiment 6: Walker-Walk + Ensemble Uncertainty")
    print("=" * 60)

    results = walker_ensemble_experiment.remote()
    print(json.dumps(results, indent=2))

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fn in ["walker_ensemble_results.json", "walker_ensemble_results.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fn, str(out / fn)], check=True,
            )
            print(f"  ✓ {fn}")
        except Exception as e:
            print(f"  skip {fn}: {e}")
    print("Done!")
