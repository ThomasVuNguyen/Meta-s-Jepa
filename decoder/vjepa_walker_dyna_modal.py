"""
V-JEPA 2 — Experiment 7: Walker-Walk Dyna Warm-Start Loop
==========================================================
Apply the Experiment 4 corrected Dyna recipe (warm-start fine-tuning from
a domain-adapted checkpoint) to `walker-walk`.

Starting from the Exp6 walker MLP checkpoint (val=0.0377), we collect
50 on-policy rollouts per round via MPC, fine-tune with LR=5e-5, and
evaluate after each round.

Hypothesis: The same -18% tip-distance improvement seen on reacher (Exp4)
should transfer to walker — 70% → 80%+ win rate over 2–3 Dyna rounds.

Compute estimate: ~100 min A10G, ~$1.85
"""

import modal
from pathlib import Path

app = modal.App("vjepa2-walker-dyna")

model_cache = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install(
        "libgl1-mesa-glx", "libglu1-mesa", "libglfw3",
        "libosmesa6", "libglew-dev", "patchelf", "xvfb",
    )
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors "
        "matplotlib Pillow numpy scipy tqdm",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,
    volumes={
        "/cache":    model_cache,
        "/output":   output_vol,
        "/rollouts": rollout_vol,
    },
)
def walker_dyna_experiment(
    n_dyna_rounds:   int = 3,          # R0=baseline eval, R1+R2=collect+FT+eval
    rollouts_per_round: int = 50,
    ep_steps:        int = 100,
    z_dim:           int = 1024,
    a_pad_dim:       int = 6,
    hidden:          int = 512,
    n_layers:        int = 3,
    ft_epochs:       int = 15,
    ft_lr:           float = 5e-5,
    batch_size:      int = 256,
    horizon:         int = 30,
    n_candidates:    int = 256,
    n_elites:        int = 32,
    n_cem_iters:     int = 5,
    n_eval_episodes: int = 10,
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

    # ─── Load walker offline data ──────────────────────────────────────────
    print("[2] Loading offline walker data (from Exp6)...")
    walker_dir = Path("/rollouts/walker_walk")
    frames_np  = np.load(str(walker_dir / "frames.npy"))
    actions_np = np.load(str(walker_dir / "actions.npy"))
    print(f"    {len(frames_np)} offline frames")

    print("[3] Embedding offline frames...")
    t0 = time.time()
    Z_offline = embed_batch(frames_np)
    print(f"    Embedded in {(time.time()-t0)/60:.1f} min")

    Z_t_off  = Z_offline[:-1]
    A_t_off  = torch.tensor(actions_np[:-1])
    Z_t1_off = Z_offline[1:]
    print(f"    Offline dataset: {len(Z_t_off)} transitions")

    # Combined dataset (offline + on-policy), grows each round
    Z_t_all  = Z_t_off.clone()
    A_t_all  = A_t_off.clone()
    Z_t1_all = Z_t1_off.clone()

    # ─── Load Exp6 MLP checkpoint (warm start) ────────────────────────────
    print("[4] Loading Exp6 walker MLP checkpoint (warm start)...")
    ckpt = torch.load("/output/walker_dynamics_mlp.pt", map_location=DEVICE, weights_only=False)
    dynamics = DynamicsMLP().to(DEVICE)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()
    print(f"    ✓ walker_dynamics_mlp.pt  val_loss={ckpt['final_val_loss']:.4f}")

    criterion = nn.MSELoss()
    rng = np.random.RandomState(55)

    def evaluate(n_episodes):
        """Run n_episodes MPC vs random, return results list."""
        from dm_control import suite
        results = []
        eval_rng = np.random.RandomState(rng.randint(0, 10000))
        for ep in range(n_episodes):
            env   = suite.load("walker", "walk", task_kwargs={"random": eval_rng.randint(0, 10000)})
            aspec = env.action_spec(); env.reset()
            mpc_reward = 0.0
            for _ in range(ep_steps):
                z_curr = embed_single(env.physics.render(256, 256, 0)).to(DEVICE)
                mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
                sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
                for _ in range(n_cem_iters):
                    eps      = eval_rng.randn(n_candidates, horizon, a_pad_dim).astype(np.float32)
                    act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
                    act_t    = torch.tensor(act_seqs, device=DEVICE)
                    z_c      = z_curr.unsqueeze(0).expand(n_candidates, -1).clone()
                    with torch.no_grad():
                        for t in range(horizon):
                            z_c = dynamics(z_c, act_t[:, t, :])
                    costs     = -z_c.norm(dim=-1).cpu().numpy()
                    elite_idx = np.argsort(costs)[:n_elites]
                    mu  = act_seqs[elite_idx].mean(axis=0)
                    sig = act_seqs[elite_idx].std(axis=0) + 1e-6
                ts = env.step(np.clip(mu[0, :len(aspec.minimum)], aspec.minimum, aspec.maximum))
                mpc_reward += float(ts.reward or 0.0)

            env2  = suite.load("walker", "walk", task_kwargs={"random": eval_rng.randint(0, 10000)})
            aspec2 = env2.action_spec(); env2.reset()
            rand_reward = 0.0
            for _ in range(ep_steps):
                ts = env2.step(eval_rng.uniform(aspec2.minimum, aspec2.maximum))
                rand_reward += float(ts.reward or 0.0)

            win = mpc_reward > rand_reward
            results.append({"mpc_reward": mpc_reward, "rand_reward": rand_reward, "win": win})
            print(f"      ep{ep+1}: MPC={mpc_reward:.1f}  rand={rand_reward:.1f}  {'✅' if win else '❌'}")
        return results

    def collect_onpolicy(n_rollouts):
        """Collect on-policy frames using current dynamics (MPC), return embedded transitions."""
        from dm_control import suite
        frames_new, actions_new = [], []
        col_rng = np.random.RandomState(rng.randint(0, 10000))
        for _ in range(n_rollouts):
            env   = suite.load("walker", "walk", task_kwargs={"random": col_rng.randint(0, 10000)})
            aspec = env.action_spec(); env.reset()
            for _ in range(ep_steps):
                frames_new.append(env.physics.render(256, 256, 0))
                z_curr = embed_single(frames_new[-1]).to(DEVICE)
                mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
                sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
                col_cem_rng = np.random.RandomState(col_rng.randint(0, 10000))
                for _ in range(3):   # fewer CEM iters during collection
                    eps      = col_cem_rng.randn(64, horizon, a_pad_dim).astype(np.float32)
                    act_seqs = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
                    act_t    = torch.tensor(act_seqs, device=DEVICE)
                    z_c      = z_curr.unsqueeze(0).expand(64, -1).clone()
                    with torch.no_grad():
                        for t in range(horizon):
                            z_c = dynamics(z_c, act_t[:, t, :])
                    costs     = -z_c.norm(dim=-1).cpu().numpy()
                    elite_idx = np.argsort(costs)[:16]
                    mu  = act_seqs[elite_idx].mean(axis=0)
                    sig = act_seqs[elite_idx].std(axis=0) + 1e-6
                a_exec = np.clip(mu[0, :len(aspec.minimum)], aspec.minimum, aspec.maximum)
                actions_new.append(pad_action(a_exec))
                env.step(a_exec)
        frames_np_new  = np.array(frames_new, dtype=np.uint8)
        actions_np_new = np.array(actions_new, dtype=np.float32)
        Z_new = embed_batch(frames_np_new)
        return Z_new[:-1], torch.tensor(actions_np_new[:-1]), Z_new[1:]

    # ─── Dyna loop ────────────────────────────────────────────────────────
    round_results = []

    # Round 0: baseline eval
    print(f"\n--- Round 0: Baseline Eval (Exp6 checkpoint) ---")
    r0_res = evaluate(n_eval_episodes)
    r0_win = float(np.mean([r["win"] for r in r0_res])) * 100
    r0_mpc = float(np.mean([r["mpc_reward"] for r in r0_res]))
    r0_rand= float(np.mean([r["rand_reward"] for r in r0_res]))
    round_results.append({
        "round": 0, "n_train": len(Z_t_all), "avg_mpc_reward": r0_mpc,
        "avg_rand_reward": r0_rand, "pct_better": r0_win,
    })
    print(f"  R0: MPC={r0_mpc:.2f}  rand={r0_rand:.2f}  win={r0_win:.0f}%")

    # Rounds 1+
    for rnd in range(1, n_dyna_rounds):
        print(f"\n--- Round {rnd}: Collect {rollouts_per_round} on-policy rollouts ---")
        t0 = time.time()
        Z_new_t, A_new_t, Z_new_t1 = collect_onpolicy(rollouts_per_round)
        n_new = len(Z_new_t)
        print(f"    Collected & embedded {n_new} transitions in {(time.time()-t0)/60:.1f} min")

        # Append to cumulative dataset
        Z_t_all  = torch.cat([Z_t_all,  Z_new_t],   dim=0)
        A_t_all  = torch.cat([A_t_all,  A_new_t],   dim=0)
        Z_t1_all = torch.cat([Z_t1_all, Z_new_t1],  dim=0)
        print(f"    Total dataset: {len(Z_t_all)} transitions")

        # Validation split
        N     = len(Z_t_all)
        perm  = np.random.RandomState(rnd).permutation(N)
        split = int(0.9 * N)
        tr, te = perm[:split], perm[split:]
        ds    = TensorDataset(Z_t_all[tr], A_t_all[tr], Z_t1_all[tr])
        dl    = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        Z_te  = Z_t_all[te].to(DEVICE)
        A_te  = A_t_all[te].to(DEVICE)
        Z1_te = Z_t1_all[te].to(DEVICE)

        print(f"    Fine-tuning for {ft_epochs} epochs (lr={ft_lr})...")
        opt = torch.optim.AdamW(dynamics.parameters(), lr=ft_lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)
        t0  = time.time()
        for epoch in range(1, ft_epochs + 1):
            dynamics.train()
            for z_b, a_b, z1_b in dl:
                z_b  = z_b.to(DEVICE); a_b = a_b.to(DEVICE); z1_b = z1_b.to(DEVICE)
                opt.zero_grad()
                criterion(dynamics(z_b, a_b), z1_b).backward()
                nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0); opt.step()
            sch.step()
        dynamics.eval()
        with torch.no_grad():
            val_loss = criterion(dynamics(Z_te, A_te), Z1_te).item()
        print(f"    FT done in {(time.time()-t0)/60:.1f} min  val_loss={val_loss:.4f}")

        # Save round checkpoint
        torch.save({
            "model_state": dynamics.state_dict(),
            "round": rnd, "val_loss": val_loss,
        }, f"/output/walker_dyna_r{rnd}.pt")
        output_vol.commit()

        print(f"    Evaluating round {rnd}...")
        r_res = evaluate(n_eval_episodes)
        r_win  = float(np.mean([r["win"]         for r in r_res])) * 100
        r_mpc  = float(np.mean([r["mpc_reward"]  for r in r_res]))
        r_rand = float(np.mean([r["rand_reward"] for r in r_res]))
        round_results.append({
            "round": rnd, "n_train": len(Z_t_all),
            "avg_mpc_reward": r_mpc, "avg_rand_reward": r_rand,
            "pct_better": r_win, "val_loss_after_ft": val_loss,
        })
        print(f"  R{rnd}: MPC={r_mpc:.2f}  rand={r_rand:.2f}  win={r_win:.0f}%  val={val_loss:.4f}")

    # ─── Chart ────────────────────────────────────────────────────────────
    rounds = [r["round"] for r in round_results]
    mpcs   = [r["avg_mpc_reward"] for r in round_results]
    rands  = [r["avg_rand_reward"] for r in round_results]
    wins   = [r["pct_better"] for r in round_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    for ax in [ax1, ax2]: ax.set_facecolor("#1a1a1a")

    ax1.plot(rounds, mpcs, "o-", color="#4fc3f7", lw=2, label="MPC reward")
    ax1.plot(rounds, rands, "s--", color="#ef5350", lw=2, label="Random reward")
    ax1.fill_between(rounds, mpcs, rands,
                     where=[m > r for m, r in zip(mpcs, rands)],
                     alpha=0.15, color="#4fc3f7", label="MPC advantage")
    ax1.set_xlabel("Dyna Round", color="white"); ax1.set_ylabel("Avg cumulative reward", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white"); ax1.set_title("Walker-Walk Dyna Rewards", color="white")

    bar_colors = ["#ef9a9a" if w < 50 else "#66bb6a" for w in wins]
    ax2.bar(rounds, wins, color=bar_colors, edgecolor="white", width=0.5)
    ax2.axhline(50, color="#ffa726", ls="--", lw=1.5, label="Random (50%)")
    for xi, w in zip(rounds, wins):
        ax2.text(xi, w + 2, f"{w:.0f}%", ha="center", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Dyna Round", color="white"); ax2.set_ylabel("Win rate (%)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white"); ax2.set_title("Walker-Walk Win Rate vs Random", color="white")

    fig.suptitle(
        f"Experiment 7: Walker-Walk Dyna Warm-Start  |  horizon T={horizon}  |  {rollouts_per_round} rollouts/round",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("/output/walker_dyna_results.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()

    final = {"experiment": 7, "env": "walker-walk", "n_dyna_rounds": n_dyna_rounds,
             "rollouts_per_round": rollouts_per_round, "ft_lr": ft_lr, "ft_epochs": ft_epochs,
             "warmstart": True, "rounds": round_results}
    with open("/output/walker_dyna_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()

    print("\n=== EXPERIMENT 7 COMPLETE ===")
    for r in round_results:
        print(f"  R{r['round']}: MPC={r['avg_mpc_reward']:.2f}  rand={r['avg_rand_reward']:.2f}  win={r['pct_better']:.0f}%")
    return final


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path

    print("=" * 60)
    print("META-S-JEPA  Experiment 7: Walker-Walk Dyna Warm-Start")
    print("=" * 60)
    results = walker_dyna_experiment.remote()
    print(json.dumps(results, indent=2))

    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fn in ["walker_dyna_results.json", "walker_dyna_results.png"]:
        try:
            subprocess.run(
                ["modal", "volume", "get", "--force",
                 "vjepa2-decoder-output", fn, str(out / fn)], check=True,
            )
            print(f"  ✓ {fn}")
        except Exception as e:
            print(f"  skip {fn}: {e}")
    print("Done!")
