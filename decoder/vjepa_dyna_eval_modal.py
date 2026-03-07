"""
Experiment 4 — Eval-only for Round 2 (+ generate final chart + JSON)
=====================================================================
R2 checkpoint is confirmed on volume (dynamics_mlp_dyna_r2.pt).
This script:
  1. Loads the R2 checkpoint
  2. Evaluates 10 episodes
  3. Generates the full 3-round comparison chart
  4. Writes corrected_dyna_results.json
"""

import modal

app = modal.App("vjepa2-dyna-eval")

model_cache = modal.Volume.from_name("vjepa2-model-cache",    create_if_missing=True)
output_vol  = modal.Volume.from_name("vjepa2-decoder-output", create_if_missing=True)
rollout_vol = modal.Volume.from_name("vjepa2-rollout-cache",  create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime")
    .apt_install("libgl1-mesa-glx","libglu1-mesa","libglfw3","libosmesa6","libglew-dev","patchelf","xvfb","ffmpeg")
    .run_commands(
        "/opt/conda/bin/pip install dm_control mujoco",
        "/opt/conda/bin/pip install transformers huggingface_hub safetensors matplotlib Pillow numpy scipy tqdm imageio[ffmpeg]",
    )
)


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={"/cache": model_cache, "/output": output_vol, "/rollouts": rollout_vol},
)
def eval_r2(n_eval: int = 10):
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import matplotlib; matplotlib.use("Agg")
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

    z_dim = 1024; a_pad_dim = 6; hidden = 512; n_layers = 3

    class DynamicsMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []; in_d = z_dim + a_pad_dim
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU()]; in_d = hidden
            layers.append(nn.Linear(in_d, z_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, z, a):
            return self.net(torch.cat([z, a], dim=-1))

    def pad_action(a, pad_to=6):
        return np.concatenate([a, np.zeros(pad_to-len(a), dtype=np.float32)])

    print("[2] Loading R2 checkpoint...")
    r2_files = sorted(Path("/output").glob("dynamics_mlp_dyna_r2*.pt"))
    ckpt = torch.load(str(r2_files[-1]), map_location=DEVICE, weights_only=False)
    dynamics = DynamicsMLP().to(DEVICE)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()
    r2_val_loss = ckpt["final_val_loss"]
    print(f"  R2 val_loss={r2_val_loss:.4f}")

    print(f"[3] Evaluating R2 ({n_eval} episodes)...")
    rng = np.random.RandomState(42)
    mpc_steps = 50; horizon = 50; n_candidates = 256; n_elites = 32; n_cem_iters = 5

    results = []
    for ep in range(n_eval):
        from dm_control import suite
        env      = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(0, 1000)})
        env_goal = suite.load("reacher", "easy", task_kwargs={"random": rng.randint(1000, 2000)})
        aspec    = env.action_spec()
        env.reset(); env_goal.reset()
        for _ in range(rng.randint(5, 25)):
            env_goal.step(rng.uniform(-1, 1, size=2))
        z_goal     = embed_single(env_goal.physics.render(height=256,width=256,camera_id=0)).to(DEVICE)
        target_pos = env_goal.physics.named.data.geom_xpos["target", :2].copy()

        for step in range(mpc_steps):
            z_curr   = embed_single(env.physics.render(height=256,width=256,camera_id=0)).to(DEVICE)
            mu  = np.zeros((horizon, a_pad_dim), dtype=np.float32)
            sig = np.ones( (horizon, a_pad_dim), dtype=np.float32)
            z_s = z_curr.unsqueeze(0).expand(n_candidates,-1)
            z_g = z_goal.unsqueeze(0).expand(n_candidates,-1)
            for _ in range(n_cem_iters):
                eps      = rng.randn(n_candidates, horizon, a_pad_dim).astype(np.float32)
                act_seqs = np.clip(mu[None]+sig[None]*eps,-1.,1.)
                act_t    = torch.tensor(act_seqs, device=DEVICE)
                z_c = z_s.clone()
                with torch.no_grad():
                    for t in range(horizon):
                        z_c = dynamics(z_c, act_t[:,t,:])
                costs     = ((z_c-z_g)**2).sum(dim=-1).cpu().numpy()
                elite_idx = np.argsort(costs)[:n_elites]
                mu  = act_seqs[elite_idx].mean(axis=0)
                sig = act_seqs[elite_idx].std(axis=0)+1e-6
            env.step(np.clip(mu[0,:2], aspec.minimum, aspec.maximum))

        mpc_dist = float(np.linalg.norm(
            env.physics.named.data.geom_xpos["finger",:2] - target_pos))
        env.reset()
        for _ in range(mpc_steps):
            env.step(rng.uniform(aspec.minimum, aspec.maximum))
        rand_dist = float(np.linalg.norm(
            env.physics.named.data.geom_xpos["finger",:2] - target_pos))

        status = "✅" if rand_dist > mpc_dist else "❌"
        print(f"  ep{ep+1}: MPC={mpc_dist:.3f}  rand={rand_dist:.3f}  {status}")
        results.append({"mpc_dist": mpc_dist, "rand_dist": rand_dist})

    avg_mpc  = float(np.mean([r["mpc_dist"]  for r in results]))
    avg_rand = float(np.mean([r["rand_dist"] for r in results]))
    win_r2   = float(np.mean([r["rand_dist"] > r["mpc_dist"] for r in results])) * 100
    print(f"\n  R2 final: MPC={avg_mpc:.3f}m  rand={avg_rand:.3f}m  win={win_r2:.0f}%")

    all_rounds = [
        {"round": 0, "n_train": 14923, "avg_mpc_dist": 0.170, "avg_random_dist": 0.184, "pct_better": 40.0},
        {"round": 1, "n_train": 17422, "avg_mpc_dist": 0.151, "avg_random_dist": 0.201, "pct_better": 70.0, "val_loss_after": 0.0144},
        {"round": 2, "n_train": 19921, "avg_mpc_dist": avg_mpc, "avg_random_dist": avg_rand, "pct_better": win_r2, "val_loss_after": r2_val_loss},
    ]

    # Chart
    rounds = [r["round"] for r in all_rounds]
    mcds   = [r["avg_mpc_dist"]    for r in all_rounds]
    rdists = [r["avg_random_dist"] for r in all_rounds]
    wins   = [r["pct_better"]      for r in all_rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    for ax in [ax1, ax2]: ax.set_facecolor("#1a1a1a")

    ax1.plot(rounds, mcds,   "o-",  color="#4fc3f7", lw=2.5, ms=9, label="MPC (corrected Dyna)")
    ax1.plot(rounds, rdists, "s--", color="#ef5350", lw=1.5, ms=7, label="Random baseline")
    ax1.axhline(0.198, color="#ffa726", ls=":", lw=1.5, label="Phase 4 FT (0.198m)")
    ax1.axhline(0.226, color="#e91e63", ls=":",  lw=1,   label="Exp3 R1 degraded (0.226m)")
    ax1.set_xlabel("Dyna Round", color="white"); ax1.set_ylabel("Avg tip→target distance (m)", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax1.set_title("Corrected Dyna: Tip Distance per Round", color="white")
    ax1.set_xticks(rounds)
    for r, d in zip(rounds, mcds):
        ax1.annotate(f"{d:.3f}m", (r, d), textcoords="offset points",
                     xytext=(0, 10), ha="center", color="#4fc3f7", fontsize=9)

    colors = ["#ef9a9a", "#66bb6a", "#a5d6a7"]
    ax2.bar(rounds, wins, color=colors[:len(rounds)], width=0.5, edgecolor="white")
    ax2.axhline(70, color="#ffa726", ls="--", lw=1.5, label="Exp3 best (70%)")
    ax2.axhline(80, color="#4fc3f7", ls="--", lw=1.5, label="Phase 4 FT (80%)")
    for r, w in zip(rounds, wins):
        ax2.text(r, w + 1.5, f"{w:.0f}%", ha="center", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Dyna Round", color="white"); ax2.set_ylabel("Win rate (%)", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax2.set_title("Corrected Dyna: Win Rate per Round", color="white")
    ax2.set_xticks(rounds)

    fig.suptitle(
        "Experiment 4: Corrected Dyna (Warm Start) — reacher-easy\n"
        "3 rounds × 50 rollouts | T=50 | warm-start Phase 4 FT → R1 → R2 | lr=5e-5",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("/output/corrected_dyna_results.png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    print("  ✓ Chart saved.")

    final = {
        "experiment": 4,
        "n_rounds": 3,
        "collect_per_round": 50,
        "ft_lr": 5e-5, "ft_epochs": 15,
        "warmstart": True,
        "rounds": all_rounds,
    }
    with open("/output/corrected_dyna_results.json", "w") as f:
        json.dump(final, f, indent=2)
    output_vol.commit()
    print("  ✓ JSON saved.")
    return final


@app.local_entrypoint()
def main():
    import subprocess, json
    from pathlib import Path
    results = eval_r2.remote()
    print(json.dumps(results, indent=2))
    out = Path("./decoder_output"); out.mkdir(exist_ok=True)
    for fn in ["corrected_dyna_results.json", "corrected_dyna_results.png"]:
        try:
            subprocess.run(["modal","volume","get","--force","vjepa2-decoder-output",fn,str(out/fn)], check=True)
            print(f"  ✓ {fn}")
        except Exception as e:
            print(f"  skipped {fn}: {e}")
