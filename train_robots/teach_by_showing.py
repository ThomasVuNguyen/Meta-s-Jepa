"""
Phase 3: Teach-by-Showing Agent
================================
Demo-conditioned agent using CEM planning + ensemble uncertainty penalty.

Pipeline:
  1. Record demo: expert plays → V-JEPA encodes frames → goal trajectory z_goal[T]
  2. Replay demo: CEM planner uses ensemble dynamics to find actions that follow z_goal
  3. Ensemble disagreement penalty prevents model exploitation (Phase 5b fix)

Usage:
  MUJOCO_GL=osmesa HF_TOKEN=xxx python3 teach_by_showing.py

Requires: torch, transformers, dm_control, mujoco, huggingface_hub
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────

TASKS = {
    "reacher_easy":      {"domain": "reacher",    "task": "easy",    "action_dim": 2, "expert": "reacher"},
    "point_mass_easy":   {"domain": "point_mass", "task": "easy",    "action_dim": 2, "expert": "point_mass"},
    "cartpole_swingup":  {"domain": "cartpole",   "task": "swingup", "action_dim": 1, "expert": "cartpole"},
}

# CEM hyperparams
CEM_POPULATION = 500      # candidate action sequences per iteration
CEM_ELITES     = 50       # top-k to refit distribution
CEM_ITERATIONS = 5        # CEM refinement iterations
CEM_HORIZON    = 8        # planning horizon (steps ahead)
UNCERTAINTY_BETA = 2.0    # ensemble disagreement penalty weight
REWARD_ALPHA   = 5.0      # predicted reward bonus weight
LOOKAHEAD      = 15       # how many demo steps ahead to track as goal
PROGRESS_THRESH = 0.8     # cosine sim threshold to advance progress

MAX_STEPS    = 200
ENSEMBLE_SIZE = 5
LATENT_DIM   = 1024
HIDDEN_DIM   = 512

BASE = Path("/root/vjepa_mvp")
MODELS = BASE / "models"
RESULTS = BASE / "results"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Expert Policies (same as overnight_v3) ───────────────────────────

EPSILON = 0.0  # no noise for demo recording

def reacher_expert(obs, aspec):
    return np.clip(obs["to_target"] * 5.0, -1.0, 1.0).astype(np.float32)

def point_mass_expert(obs, aspec):
    pos = obs.get("position", np.zeros(2))
    vel = obs.get("velocity", np.zeros(2))
    return np.clip(-2.0 * pos - 0.5 * vel, aspec.minimum, aspec.maximum).astype(np.float32)

def cartpole_expert(obs, aspec):
    pos = obs.get("position", np.zeros(3))
    vel = obs.get("velocity", np.zeros(2))
    cos_a = pos[1] if len(pos) > 1 else 1.0
    sin_a = pos[2] if len(pos) > 2 else 0.0
    cart_pos, cart_vel = pos[0], vel[0] if len(vel) > 0 else 0
    pole_vel = vel[1] if len(vel) > 1 else 0
    angle = np.arctan2(sin_a, cos_a)
    energy = 0.5 * pole_vel**2 + 9.81 * (cos_a - 1)
    if abs(angle) < 0.3:
        a = 5.0*angle + 1.0*pole_vel - 2.0*cart_pos - 1.0*cart_vel
    else:
        a = 3.0*pole_vel*cos_a + 0.5*energy*np.sign(pole_vel)
    return np.clip(np.array([a]), aspec.minimum, aspec.maximum).astype(np.float32)

EXPERTS = {"reacher": reacher_expert, "point_mass": point_mass_expert, "cartpole": cartpole_expert}


# ── Models ───────────────────────────────────────────────────────────

class DynamicsModel(nn.Module):
    def __init__(self, ldim=LATENT_DIM, adim=2, hdim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim + adim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Linear(hdim, ldim))
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))

class RewardModel(nn.Module):
    def __init__(self, ldim=LATENT_DIM, adim=2, hdim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim + adim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, 1))
    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)


# ── V-JEPA Encoder ──────────────────────────────────────────────────

class VJEPAEncoder:
    """Wraps V-JEPA 2 for frame encoding with sliding window."""

    def __init__(self, device="cuda"):
        from transformers import AutoModel
        from torchvision import transforms
        self.device = device
        log("Loading V-JEPA 2 ViT-L...")
        cache = str(BASE / "cache/hf")
        os.makedirs(cache, exist_ok=True)
        self.model = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True,
            cache_dir=cache).to(device, dtype=torch.float16).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.frame_buffer = []
        log("V-JEPA 2 ready.")

    def reset(self):
        self.frame_buffer = []

    def encode_frame(self, frame_rgb):
        """Encode a single frame using 8-frame sliding window."""
        from PIL import Image
        ft = self.transform(Image.fromarray(frame_rgb))
        self.frame_buffer.append(ft)

        # Build 8-frame window
        window = self.frame_buffer[-8:]
        while len(window) < 8:
            window = [window[0]] + window
        clip = torch.stack(window).unsqueeze(0).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            out = self.model(pixel_values_videos=clip)
            emb = out.last_hidden_state.mean(dim=1).cpu().float().squeeze(0).numpy()
        return emb

    def encode_frames_batch(self, frames):
        """Encode multiple frames, returns list of embeddings."""
        self.reset()
        embs = []
        for f in frames:
            embs.append(self.encode_frame(f))
        return np.array(embs)


# ── Ensemble Dynamics ────────────────────────────────────────────────

class EnsembleDynamics:
    """5× MLP ensemble for uncertainty-aware prediction."""

    def __init__(self, task_name, action_dim, device="cuda"):
        self.device = device
        self.models = []
        self.action_dim = action_dim

        md = MODELS / task_name
        for i in range(ENSEMBLE_SIZE):
            m = DynamicsModel(adim=action_dim).to(device)
            m.load_state_dict(torch.load(str(md / f"dyn_{i}.pt"), map_location=device, weights_only=True))
            m.eval()
            self.models.append(m)

        self.reward_model = RewardModel(adim=action_dim).to(device)
        self.reward_model.load_state_dict(
            torch.load(str(md / "reward.pt"), map_location=device, weights_only=True))
        self.reward_model.eval()
        log(f"  Loaded {ENSEMBLE_SIZE}× dynamics + reward for {task_name}")

    @torch.no_grad()
    def predict(self, z, a):
        """Predict next state with all ensemble members.
        z: (B, latent_dim), a: (B, action_dim)
        Returns: mean_z_next (B, latent_dim), std_z_next (B, latent_dim), preds list
        """
        preds = []
        for m in self.models:
            preds.append(m(z, a))
        stacked = torch.stack(preds)       # (E, B, D)
        mean = stacked.mean(dim=0)          # (B, D)
        std = stacked.std(dim=0)            # (B, D)
        return mean, std, preds

    @torch.no_grad()
    def predict_reward(self, z, a):
        """Predict reward. z: (B, D), a: (B, A) → (B,)"""
        return self.reward_model(z, a)

    def disagreement(self, std):
        """Ensemble disagreement = mean std across latent dims. (B,)"""
        return std.mean(dim=-1)


# ── CEM Planner ──────────────────────────────────────────────────────

class CEMPlanner:
    """Cross-Entropy Method planner with ensemble uncertainty penalty."""

    def __init__(self, ensemble, action_dim, device="cuda"):
        self.ensemble = ensemble
        self.action_dim = action_dim
        self.device = device

    def plan(self, z_current, z_goals, progress_idx):
        """Plan the best action using CEM.

        Args:
            z_current: current latent state (latent_dim,)
            z_goals: full goal trajectory (T, latent_dim)
            progress_idx: current progress along the goal trajectory

        Returns:
            best_action: (action_dim,) numpy array
            info: dict with planning stats
        """
        B = CEM_POPULATION
        H = CEM_HORIZON
        A = self.action_dim
        D = LATENT_DIM

        # Goal: look ahead from current progress
        goal_indices = []
        for h in range(H):
            idx = min(progress_idx + h + 1, len(z_goals) - 1)
            goal_indices.append(idx)
        z_goal_seq = torch.tensor(z_goals[goal_indices], dtype=torch.float32, device=self.device)  # (H, D)

        z0 = torch.tensor(z_current, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1)  # (B, D)

        # Initialize action distribution
        mu = torch.zeros(H, A, device=self.device)
        sigma = torch.ones(H, A, device=self.device) * 0.5

        best_action = None
        best_score = -float("inf")

        for iteration in range(CEM_ITERATIONS):
            # Sample action sequences: (B, H, A)
            noise = torch.randn(B, H, A, device=self.device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            actions = actions.clamp(-1.0, 1.0)

            # Rollout through ensemble dynamics
            total_score = torch.zeros(B, device=self.device)
            z = z0.clone()

            for h in range(H):
                a_h = actions[:, h, :]  # (B, A)

                # Predict next state with uncertainty
                z_next_mean, z_next_std, _ = self.ensemble.predict(z, a_h)
                disagreement = self.ensemble.disagreement(z_next_std)  # (B,)

                # Predict reward
                pred_reward = self.ensemble.predict_reward(z, a_h)  # (B,)

                # Goal-following score: negative L2 distance to goal at this timestep
                goal_dist = torch.norm(z_next_mean - z_goal_seq[h].unsqueeze(0), dim=-1)  # (B,)

                # Combined score:
                #   + predicted reward (encourage high-reward states)
                #   - goal distance (follow the demo)
                #   - uncertainty penalty (avoid regions where ensemble disagrees)
                step_score = (
                    REWARD_ALPHA * pred_reward
                    - goal_dist
                    - UNCERTAINTY_BETA * disagreement
                )
                total_score += step_score

                z = z_next_mean  # advance using mean prediction

            # Select elites
            elite_idx = total_score.topk(CEM_ELITES).indices
            elite_actions = actions[elite_idx]  # (K, H, A)

            # Refit distribution
            mu = elite_actions.mean(dim=0)    # (H, A)
            sigma = elite_actions.std(dim=0).clamp(min=0.05)  # (H, A)

            # Track best
            iter_best = total_score[elite_idx[0]].item()
            if iter_best > best_score:
                best_score = iter_best
                best_action = elite_actions[0, 0].cpu().numpy()  # first action of best sequence

        info = {
            "score": best_score,
            "mean_disagreement": self.ensemble.disagreement(z_next_std).mean().item(),
            "goal_dist": goal_dist[elite_idx[0]].item(),
        }
        return best_action, info


# ── Teach-by-Showing Agent ──────────────────────────────────────────

class TeachByShowingAgent:
    """Watch a demo, then replay it using CEM + ensemble uncertainty."""

    def __init__(self, task_name, encoder, device="cuda"):
        self.task_name = task_name
        self.cfg = TASKS[task_name]
        self.encoder = encoder
        self.device = device

        # Load ensemble
        self.ensemble = EnsembleDynamics(task_name, self.cfg["action_dim"], device)
        self.planner = CEMPlanner(self.ensemble, self.cfg["action_dim"], device)

        self.demo_z = None
        self.demo_rewards = None
        self.demo_actions = None

    def record_demo(self, seed=42, n_steps=MAX_STEPS):
        """Record a demo using the expert policy."""
        from dm_control import suite

        log(f"  Recording demo (seed={seed})...")
        env = suite.load(self.cfg["domain"], self.cfg["task"], task_kwargs={"random": seed})
        expert = EXPERTS[self.cfg["expert"]]
        aspec = env.action_spec()
        ts = env.reset()
        obs = ts.observation

        frames = [env.physics.render(height=224, width=224, camera_id=0).copy()]
        actions = []
        rewards = []

        for step in range(n_steps):
            a = expert(obs, aspec)
            actions.append(a.copy())
            ts = env.step(a)
            obs = ts.observation
            rewards.append(float(ts.reward or 0.0))
            frames.append(env.physics.render(height=224, width=224, camera_id=0).copy())

        # Encode demo frames
        log(f"  Encoding demo ({len(frames)} frames)...")
        self.demo_z = self.encoder.encode_frames_batch(frames)
        self.demo_rewards = np.array(rewards)
        self.demo_actions = np.array(actions)
        self.demo_frames = frames

        demo_return = sum(rewards)
        log(f"  Demo recorded: return={demo_return:.1f}, "
            f"success={'YES' if demo_return > 100 else 'NO'}")
        return demo_return

    def replay_demo(self, seed=42, n_steps=MAX_STEPS, shifted_seed=None):
        """Replay the demo using CEM planning.

        Args:
            seed: if shifted_seed is None, use same seed as demo
            shifted_seed: if set, use different seed (shifted start test)
        """
        from dm_control import suite

        actual_seed = shifted_seed if shifted_seed is not None else seed
        condition = "shifted" if shifted_seed else "faithful"
        log(f"  Replaying ({condition}, seed={actual_seed})...")

        env = suite.load(self.cfg["domain"], self.cfg["task"],
                        task_kwargs={"random": actual_seed})
        ts = env.reset()
        self.encoder.reset()

        # Get initial frame and encode
        frame = env.physics.render(height=224, width=224, camera_id=0).copy()
        z_current = self.encoder.encode_frame(frame)

        # Track progress along demo trajectory
        progress = 0
        replay_actions = []
        replay_rewards = []
        replay_frames = [frame]
        replay_z = [z_current]
        planning_info = []

        for step in range(n_steps):
            # Plan action using CEM
            action, info = self.planner.plan(z_current, self.demo_z, progress)
            replay_actions.append(action.copy())
            planning_info.append(info)

            # Execute action
            ts = env.step(action)
            reward = float(ts.reward or 0.0)
            replay_rewards.append(reward)

            # Observe result
            frame = env.physics.render(height=224, width=224, camera_id=0).copy()
            z_current = self.encoder.encode_frame(frame)
            replay_frames.append(frame)
            replay_z.append(z_current)

            # Update progress (advance when close to next goal)
            if progress < len(self.demo_z) - 1:
                z_goal = self.demo_z[progress + 1]
                cos_sim = np.dot(z_current, z_goal) / (
                    np.linalg.norm(z_current) * np.linalg.norm(z_goal) + 1e-8)
                if cos_sim > PROGRESS_THRESH:
                    progress += 1

            if (step + 1) % 50 == 0 or step == 0:
                log(f"    Step {step+1:3d}/{n_steps} | R={sum(replay_rewards):.1f} | "
                    f"Progress={progress}/{len(self.demo_z)} | "
                    f"Disagree={info['mean_disagreement']:.4f} | "
                    f"GoalDist={info['goal_dist']:.2f}")

        replay_return = sum(replay_rewards)
        log(f"  Replay done: return={replay_return:.1f}, "
            f"progress={progress}/{len(self.demo_z)}")

        return {
            "condition": condition,
            "seed": actual_seed,
            "return": replay_return,
            "progress": progress,
            "max_progress": len(self.demo_z),
            "actions": np.array(replay_actions),
            "rewards": np.array(replay_rewards),
            "frames": replay_frames,
            "z": np.array(replay_z),
            "planning_info": planning_info,
        }


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_task(task_name, encoder, device="cuda", n_demos=5):
    """Full evaluation of a task: multiple demo seeds × faithful + shifted replay."""
    log(f"\n{'='*60}")
    log(f"EVALUATING: {task_name}")
    log(f"{'='*60}")

    agent = TeachByShowingAgent(task_name, encoder, device)
    results = []

    for demo_seed in range(n_demos):
        log(f"\n--- Demo {demo_seed+1}/{n_demos} (seed={demo_seed}) ---")

        # Record demo
        demo_return = agent.record_demo(seed=demo_seed)

        # Faithful replay (same seed)
        faithful = agent.replay_demo(seed=demo_seed)
        faithful["demo_return"] = demo_return
        faithful["demo_seed"] = demo_seed
        results.append(faithful)

        # Shifted replay (different seed)
        shifted = agent.replay_demo(seed=demo_seed, shifted_seed=demo_seed + 1000)
        shifted["demo_return"] = demo_return
        shifted["demo_seed"] = demo_seed
        results.append(shifted)

    return results


def run_evaluation(tasks=None, n_demos=3):
    """Run full evaluation across tasks."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    RESULTS.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    encoder = VJEPAEncoder(device)

    if tasks is None:
        tasks = list(TASKS.keys())

    all_results = {}
    for task_name in tasks:
        task_results = evaluate_task(task_name, encoder, device, n_demos)
        all_results[task_name] = task_results

    # Print summary table
    log(f"\n{'='*70}")
    log("RESULTS SUMMARY")
    log(f"{'='*70}")
    log(f"{'Task':<20} {'Condition':<12} {'Demo R':>8} {'Replay R':>10} {'Ratio':>8} {'Progress':>10}")
    log("-" * 70)

    for task_name, results in all_results.items():
        for r in results:
            ratio = r["return"] / max(r["demo_return"], 0.01)
            log(f"{task_name:<20} {r['condition']:<12} "
                f"{r['demo_return']:>8.1f} {r['return']:>10.1f} "
                f"{ratio:>7.1%} "
                f"{r['progress']:>4d}/{r['max_progress']}")

    # Summary stats
    log(f"\n{'='*70}")
    for task_name, results in all_results.items():
        faithful = [r for r in results if r["condition"] == "faithful"]
        shifted = [r for r in results if r["condition"] == "shifted"]

        if faithful:
            avg_fr = np.mean([r["return"] for r in faithful])
            avg_dr = np.mean([r["demo_return"] for r in faithful])
            log(f"  {task_name} faithful: avg replay={avg_fr:.1f} / demo={avg_dr:.1f} "
                f"({avg_fr/max(avg_dr, 0.01):.0%})")
        if shifted:
            avg_sr = np.mean([r["return"] for r in shifted])
            log(f"  {task_name} shifted:  avg replay={avg_sr:.1f}")

    # Save results
    save_results = {}
    for task_name, results in all_results.items():
        save_results[task_name] = [{
            "condition": r["condition"],
            "seed": int(r["seed"]),
            "demo_seed": int(r["demo_seed"]),
            "return": float(r["return"]),
            "demo_return": float(r["demo_return"]),
            "progress": int(r["progress"]),
            "max_progress": int(r["max_progress"]),
            "avg_disagreement": float(np.mean([i["mean_disagreement"] for i in r["planning_info"]])),
            "avg_goal_dist": float(np.mean([i["goal_dist"] for i in r["planning_info"]])),
        } for r in results]

    with open(str(RESULTS / "eval_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    log(f"\nResults saved to {RESULTS / 'eval_results.json'}")

    return all_results


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Tasks to evaluate (default: all)")
    parser.add_argument("--n-demos", type=int, default=3,
                       help="Number of demo seeds per task")
    parser.add_argument("--reacher-only", action="store_true",
                       help="Only run reacher_easy (quick test)")
    args = parser.parse_args()

    tasks = ["reacher_easy"] if args.reacher_only else args.tasks
    run_evaluation(tasks=tasks, n_demos=args.n_demos)
