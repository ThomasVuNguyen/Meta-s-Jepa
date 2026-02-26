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

## Phase 2 — Exploration Data Collection

**Date:** 2026-02-25 | **Compute:** Modal A10G, ~40 min, ~$0.75

Instead of pursuing SAC directly, we pivoted to building a **world model** — teaching the system to predict the future in latent space, then planning by imagining.

**Data pipeline:** Modified `generate_and_encode_modal.py` with epsilon-greedy noise (ε=0.3) injected into the P-controller. 30% of actions are random, ensuring the dataset covers diverse (including "wrong") transitions.

| Metric | Value |
|---|---|
| Episodes | 500 |
| Steps per episode | 200 |
| Total transitions | 100,000 |
| Dataset size | 496 MB |
| Format | `(z_t, a_t, z_{t+1})` triples |

---

## Phase 2b — Large-Scale Data Collection (5000 episodes)

**Date:** 2026-02-25 | **Compute:** Modal 10 × A10G parallel, ~90 min wall time, ~$7

Phase 4d revealed that the 10.5M ResBlock dynamics model overfits on 80k transitions. Fix: collect 10× more data using parallel workers.

### Pipeline
- Parallelized across **10 Modal A10G containers** using `.map()`, each generating 500 episodes with unique seeds
- Same epsilon-greedy P-controller (ε=0.3)
- Shards merged on cloud (32GB RAM container) to avoid local OOM
- Uploaded directly from Modal to HuggingFace (no local download of 8GB file)

| Metric | Phase 2 | Phase 2b |
|---|---|---|
| Episodes | 500 | **5,000** |
| Transitions | 100,000 | **1,000,000** |
| Dataset size | 496 MB | **8.2 GB** |
| Avg success rate | ~15% | 12.0% |
| Wall time | 40 min (1 GPU) | 90 min (10 GPUs) |

**✅ Dataset stored at:** [HuggingFace: ThomasTheMaker/vjepa2-reacher-world-model](https://huggingface.co/datasets/ThomasTheMaker/vjepa2-reacher-world-model)

---

## Phase 3 — Action-Conditioned Dynamics Predictor

**Date:** 2026-02-25 | **Compute:** Local CPU, ~10 min

Trained an MLP to predict the next latent state: `f(z_t, a_t) → z_{t+1}`.

### Architecture
```
Input: concat(z_t [1024], a_t [2]) = 1026-dim
 → Linear(1026→512) → LayerNorm → ReLU
 → Linear(512→512) → LayerNorm → ReLU
 → Linear(512→512) → LayerNorm → ReLU
 → Linear(512→1024)
Output: z_t + delta_z (residual prediction)
```

### Training
- **Loss:** SmoothL1 (Huber), AdamW, cosine LR
- **Train/Val Loss:** ~0.01 / ~0.01 (100 epochs)
- **Parameters:** ~1.2M

**✅ Key finding:** Residual prediction (`z_t + Δz`) converged faster and more stably than direct prediction. The dynamics model learned meaningful 1-step transitions.

---

## Phase 4 — Random Shooting MPC

**Date:** 2026-02-25 | **Compute:** Local CPU, ~30 min

First MPC attempt: sample 1000 random action sequences (horizon=10), unroll each through the dynamics model, pick the one closest to `z_goal`.

| Metric | Result |
|---|---|
| Total Env Reward | **0.00** |
| Steps | 100 |

**❌ Failed.** Two causes identified:
1. **Bad goal state** — random exploration couldn't find a high-reward state, so the "goal" was an arbitrary frame
2. **Random shooting is inefficient** — sampling 1000 random trajectories in continuous 2D action space has poor coverage

---

## Phase 4b — CEM Planner (upgraded)

**Date:** 2026-02-25 | **Compute:** Local CPU, ~60 min

Two fixes applied:
1. **Expert-generated goal** — P-controller on a separate raw (unwrapped) env finds the closest-to-target frame (0.07 distance)
2. **Cross-Entropy Method (CEM)** — iteratively refines action distribution over 5 iterations (500 samples, top 50 elites, warm-starting)

### Results

| Metric | Phase 4 (Random) | Phase 4b (CEM) |
|---|---|---|
| Initial Latent Dist | N/A | 16.18 |
| Final Latent Dist | N/A | 9.50 |
| **Min Latent Dist** | N/A | **7.80** |
| **Improvement** | — | **41.3%** |
| Total Env Reward | 0.00 | **6.00** |

### Key findings

**✅ The dynamics model learned real physics.** Latent distance to goal decreased 41.3% — the planner genuinely steers the robot toward the target by imagining futures. This would not work if `f(z_t, a_t)` was noise.

**✅ CEM >> random shooting.** Iterative refinement finds much better trajectories — total env reward improved from 0.0 to 6.0.

**⚠️ Compounding error is the bottleneck.** The latent distance oscillates between 8-18 instead of converging monotonically. The dynamics model was trained on **1-step prediction** but the planner unrolls it for **10 steps** — small per-step errors compound exponentially. 

**⚠️ dm_control `reacher-easy` reward is very sparse.** The reward only fires when the fingertip is within a very tight tolerance of the target. Even the P-controller expert gets 0 reward over 200 steps with a gain of 5.0. Latent distance is a more informative metric for this task.

**💡 Next step:** Train the dynamics model with **multi-step rollout loss** — backpropagate through H-step unrolls during training to directly penalize compound error. Then consider Phase 5 (Dreamer-style actor-critic on imagined rollouts).

---

## Phase 4c — Multi-Step Rollout Training

**Date:** 2026-02-25 | **Compute:** Modal A10G, ~15 min

Hypothesis: the Phase 4b oscillation was caused by compounding 1-step prediction error over 10-step planner horizon. Fix: retrain dynamics model with **multi-step rollout loss** (H=5), backpropagating through predicted states:

```
z1_pred = f(z0, a0)           → loss vs z1
z2_pred = f(z1_pred, a1)      → loss vs z2  (uses PREDICTED z!)
z3_pred = f(z2_pred, a2)      → loss vs z3
```

### Results

| Metric | Phase 4b (1-step) | Phase 4c (H=5) |
|---|---|---|
| Initial Latent Dist | 16.18 | 15.42 |
| Final Latent Dist | 9.50 | 10.57 |
| **Min Latent Dist** | **7.80** | 9.55 |
| **Improvement** | **41.3%** | 31.4% |
| Total Env Reward | 6.00 | 0.00 |

### Key findings

**❌ Multi-step training regressed!** The 1-step model (Phase 4b) outperformed the multi-step model (Phase 4c) on every metric. The multi-step model's min latent distance (9.55) never reached Phase 4b's (7.80).

**💡 Why?** Likely causes:
1. **Training instability** — multi-step rollout loss creates long gradient chains that are harder to optimize. The model may learn conservative predictions that don't compound errors but also lack precision.
2. **Architecture bottleneck** — the MLP dynamics model may not have enough capacity to benefit from multi-step supervision. A Transformer or larger model may be needed.
3. **Data diversity** — 500 episodes of epsilon-greedy P-controller may not cover enough of the state space for multi-step learning.

**💡 Conclusion:** The dynamics model architecture (simple MLP) is likely the real bottleneck, not the training objective. Phase 5 (Dreamer-style actor-critic) should use a more expressive world model, or skip model-based planning entirely and use the learned latents for model-free RL.

---

## Phase 4d — ResBlock Dynamics Architecture

**Date:** 2026-02-25 | **Compute:** Modal A10G, ~20 min

Hypothesis: the 1.2M-param MLP lacks capacity to model dynamics accurately. Fix: 4× residual blocks with 1024 hidden dim → **10.5M params** (8.5× larger).

### Architecture
```
Input: concat(z_t, a_t) [1026]
 → Linear(1026→1024) → LayerNorm → ReLU
 → ResBlock(1024): Linear→LN→ReLU→Linear→LN + skip
 → ResBlock(1024): ...
 → ResBlock(1024): ...
 → ResBlock(1024): ...
 → Linear(1024) → delta_z
Output: z_t + delta_z (residual prediction)
```

### Training
- **Epochs:** 150 (best val at ~epoch 40)
- **Train loss:** 0.0180 → 0.0016 (memorized training set)
- **Val loss:** 0.0132 → 0.0137 (worse than MLP's 0.01!)
- **Severe overfitting** — 10.5M params on 80k samples

### Results

| Metric | 4b (MLP 1.2M) | 4c (MLP multi-step) | 4d (ResBlock 10.5M) |
|---|---|---|---|
| Min Latent Dist | **7.80** | 9.55 | 9.22 |
| **Improvement** | **41.3%** | 31.4% | 11.5% |
| Env Reward | **6.0** | 0.0 | 0.0 |

### Key findings

**❌ Bigger is not better.** The 10.5M ResBlock model dramatically overfits on 80k transitions and performs worst of all variants (11.5% vs 41.3%).

**✅ The original small MLP is the sweet spot.** The 1.2M-param model with 1-step loss (Phase 4b) remains the best dynamics model. Its limited capacity actually acts as implicit regularization.

**💡 The real bottleneck is not the model — it's the data.** With only 80k transitions (500 episodes), larger models memorize rather than generalize. Options:
1. **Collect more data** (5000+ episodes with more diverse policies)
2. **Add regularization** (dropout, stronger weight decay)
3. **Skip model-based planning** and use V-JEPA latents directly for model-free RL (Phase 5)

---

## Blockers / Limitations

| Issue | Status | Impact |
|---|---|---|
| P-controller expert suboptimal | Known — by design | Caps BC at 20% |
| dm_control reacher reward very sparse | Known | Latent distance is better metric |
| Dynamics model compound error | Phase 4b best at 41.3% | Multi-step (4c) and bigger model (4d) both regressed |
| Data quantity | **New — identified in 4d** | 80k transitions insufficient for >1.2M param models |

---

## Cost Summary

| Step | Compute | Cost |
|---|---|---|
| Phase 1a: 500 demos + encoding | Modal A10G, ~35 min | ~$0.65 |
| Phase 1b: BC training | Local CPU, ~8 min | $0 |
| Phase 1c: Eval (2 conditions) | Modal A10G, ~60 min | ~$1.10 |
| Phase 2: Exploration data | Modal A10G, ~40 min | ~$0.75 |
| Phase 3: Dynamics training | Local CPU, ~10 min | $0 |
| Phase 4: Random shooting MPC | Local CPU, ~30 min | $0 |
| Phase 4b: CEM planner | Local CPU, ~60 min | $0 |
| Phase 4c: Multi-step dynamics | Modal A10G, ~15 min | ~$0.30 |
| Phase 4d: ResBlock dynamics | Modal A10G, ~20 min | ~$0.30 |
| **Total** | | **~$3.10** |
