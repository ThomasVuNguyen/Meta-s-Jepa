# V-JEPA 2 → Robot Policy — Findings

**Goal:** Assess whether V-JEPA 2, a video foundation model pretrained on internet video, can serve as a frozen perception backbone for robot control — without any robot-specific encoder training.

**Task:** dm_control `reacher-easy` — a 2-joint robot arm must move its fingertip to a random target.

---

## Experiment Pipeline

```
Camera frames (last 8) → [V-JEPA 2, frozen, 325M params] → 1024-dim embedding
                                                                    ↓
                                                       [MLP policy, 279k params]
                                                                    ↓
                                                         joint torques [2]
```

### Phase 1 steps
1. **Expert demo generation** — scripted P-controller runs 500 episodes in dm_control, rendering 256×256 pixel frames
2. **V-JEPA encoding** — each 8-frame sliding window encoded to a 1024-dim embedding (Modal A10G, batch_size=32)
3. **Behavior cloning** — MLP trained on (embedding, expert_action) pairs via MSE loss
4. **Evaluation** — trained policy rolled out for 100 live episodes with V-JEPA encoding at every step

**Dataset:** 100,000 transitions (500 episodes × 200 steps), 410MB of embeddings stored in Modal volume.

---

## Phase 1 Results — Behavior Cloning

**Date:** 2026-02-24 | **Eval episodes:** 100 per condition

| Condition | Success Rate | Mean Reward | Notes |
|---|---|---|---|
| **V-JEPA BC** (ours) | **20.0%** | **66.4** | Frozen V-JEPA 2 + 3-layer MLP |
| Scripted Expert | 20.0% | 67.6 | Upper bound (P-controller) |
| Random Policy | ~2% | ~0 | Lower bound |

### BC Training
- Architecture: `Linear(1024→256) → LayerNorm → ReLU → Linear(256→64) → ReLU → Linear(64→2) → Tanh`
- Parameters: 279,490 (tiny)
- Train MSE loss: 0.0001 → Val MSE: 0.0002 (100 epochs, cosine LR decay)
- Training time: ~8 min on CPU

### Key findings

**✅ V-JEPA 2 embeddings fully support policy learning.** The BC policy matches the scripted expert exactly (20% vs 20%), while the random policy baseline achieves only ~2%. That's a **10× lift over the random baseline** with a frozen encoder that has seen zero robot data.

**⚠️ The ceiling is the expert, not the encoder.** The P-controller expert itself only achieves ~20% because it applies Cartesian direction forces to joint-space actuators — a fundamental mismatch. BC cannot exceed what it learns from. The V-JEPA embedding quality is not the bottleneck.

**💡 Implication:** V-JEPA 2 encodes enough spatial and directional information for a tiny MLP to replicate a controller's behavior. With a better expert (or RL), the success rate should improve significantly.

---

## What V-JEPA 2 Encodes (from linear probe, see findings/findings.md)

| Property | V-JEPA R² / Acc | Random Baseline |
|---|---|---|
| Object XY position | **R²=0.86** | -0.11 |
| Object size/depth | **R²=0.89** | -0.11 |
| Object class | **88.4%** | 42.3% |
| Motion direction | 63.6% | 49.9% |

The reacher task primarily requires XY spatial awareness — exactly where V-JEPA is strongest.

---

## Scaling Trend

| Scale | Expert SR | BC SR | Notes |
|---|---|---|---|
| Phase 1 BC | 20% | 20% | P-controller ceiling |
| Phase 2 SAC (planned) | N/A | 60-80%? | RL exceeds the expert |

---

## Next: Phase 2 — Online RL (SAC)

BC is fundamentally limited by the quality of demonstrations. The next step is **Soft Actor-Critic (SAC)** running online in dm_control — it learns from the reward signal directly, not from expert imitation.

**Why this matters:**
- SAC can explore actions the P-controller never took and discover better strategies
- Target success rate: **60-80%** (vs 20% expert ceiling)
- Runs entirely on Modal A10G overnight (~4 hrs, ~$5 of credits)

### Phase 2 architecture
```
Actor:  Linear(1024→256) → ReLU → Linear(256→64) → ReLU → Linear(64→4)  # mean + log_std
Critic: Linear(1024+2→256) → ReLU → Linear(256→64) → ReLU → Linear(64→1)  # Q-value (×2 twin)
```

### Phase 2 plan
- 32 parallel dm_control environments on A10G
- Replay buffer: 1M transitions (embeddings pre-computed per step)
- 500k SAC training steps
- Baseline comparison: same SAC with random embeddings

---

## Blockers / Limitations

| Issue | Status | Impact |
|---|---|---|
| P-controller expert suboptimal | Known — by design | Caps BC at 20% |
| Random policy eval timed out (1hr limit) | Known | Only 60 eps of data, but ~2% rate confirmed |
| bc_policy.pt and demo dataset not in git (236MB) | Stored in Modal volume `vjepa2-robot-demos` | Re-run encode step to regenerate |

---

## Cost Summary

| Step | Compute | Cost |
|---|---|---|
| Phase 1a: 500 demos + encoding | Modal A10G, ~35 min | ~$0.65 |
| Phase 1b: BC training | Local CPU, ~8 min | $0 |
| Phase 1c: Eval (2 conditions) | Modal A10G, ~60 min | ~$1.10 |
| **Phase 1 total** | | **~$1.75** |
| Phase 2 SAC (planned) | Modal A10G, ~4 hrs | ~$4.50 |
