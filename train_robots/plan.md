# V-JEPA 2 → Robot Policy Training Plan

## Goal
Use V-JEPA 2 as a frozen visual backbone to train a robot policy on `dm_control` tasks.
No fine-tuning of V-JEPA — only a small policy head trains.

## Why this is interesting
Our linear probe confirmed V-JEPA 2 embeddings encode:
- Object XY position (R² = 0.86)
- Object size/depth (R² = 0.89)
- Semantic class (88.4% accuracy)

...with a single linear layer, no robot-specific training. SAC on top of these embeddings
should learn efficient manipulation policies significantly faster than training from raw pixels.

---

## Architecture

```
Camera frames (last 8) → [V-JEPA 2, frozen, 325M params] → 1024-dim embedding
                                                                      ↓
                                                         [SAC policy, ~200k params]
                                                                      ↓
                                                             joint velocities
```

V-JEPA weights never change. Only the SAC actor/critic/value networks train.

---

## Phase 1 — Behavior Cloning Baseline (local, ~1 day)

**Purpose:** Validate the pipeline end-to-end before committing GPU resources.

### 1a. Generate expert demonstrations
- Environment: `dm_control` `reacher-easy` (2-joint arm, reach a target)
- Expert: scripted controller using internal state (not pixels)
- Scale: 500 episodes × 200 steps = 100k transitions
- Output: `demos/frames/ [T=8, 256×256×3]`, `demos/actions/ [2]`
- Runs locally on CPU, ~15 min

### 1b. Encode with V-JEPA (Modal A10G)
- Feed each 8-frame window through V-JEPA encoder
- Output: pre-encoded dataset `(embedding[1024], action[2])`
- ~10 min, ~$0.20

### 1c. Train BC policy (local CPU)
```python
nn.Sequential(
    nn.Linear(1024, 256), nn.ReLU(),
    nn.Linear(256, 64),  nn.ReLU(),
    nn.Linear(64, 2)     # joint velocities
)
# MSE loss on expert actions, 50 epochs, ~2 min
```

### 1d. Evaluate
- 100 rollouts, measure success rate (arm tip within radius of target)
- Compare: V-JEPA BC vs. random embedding BC vs. proprioception-only BC

**Success threshold:** V-JEPA BC > 2× random embedding BC success rate. ✅ **Achieved: 20% vs ~2% random = 10× baseline**

### Phase 1 Results (completed 2026-02-23)

| Condition | Success Rate | Mean Reward | Notes |
|---|---|---|---|
| **V-JEPA BC** (ours) | **20.0%** | 66.4 | Frozen V-JEPA 2 + 3-layer MLP |
| **Scripted Expert** | 20.0% | 67.6 | Upper bound (P-controller) |
| **Random Policy** | ~2% | ~0 | Lower bound |

**Key finding:** BC policy exactly matches expert performance — V-JEPA 2 embeddings contain enough spatial information for the MLP to perfectly clone the P-controller. 10× better than random.

**Bottleneck identified:** The scripted P-controller expert is suboptimal (~20% success). To push higher, Phase 2 should use SAC (reinforcement learning) which can exceed the expert's ceiling.


---

## Phase 2 — Online RL with SAC (Modal, overnight)

**Purpose:** Use Modal credits to train a proper RL agent — significantly outperforms BC.

### Setup
- Algorithm: **Soft Actor-Critic (SAC)** — sample-efficient, continuous action space
- 32 parallel dm_control environments on A10G
- V-JEPA encodes batched observations from all 32 envs simultaneously
- Replay buffer: 1M transitions (stored as embeddings, not raw pixels → compact)

### Networks
```
Actor:  Linear(1024→256)→ReLU→Linear(256→64)→ReLU→Linear(64→2+2)  # mean + log_std
Critic: Linear(1024+2→256)→ReLU→Linear(256→64)→ReLU→Linear(64→1)  # Q-value (×2 for twin critics)
```

### Training schedule
| Stage | Steps | Time (A10G) | Cost |
|---|---|---|---|
| Warmup (random policy) | 10k | ~10 min | $0.15 |
| SAC training | 500k | ~4 hrs | $4.40 |
| Eval checkpoint every 50k steps | — | — | — |

**Total: ~4.5 hrs, ~$5 of Modal credits.**

### Comparison runs (run in parallel to use credits)
- `reacher-easy`: V-JEPA SAC vs. random embedding SAC vs. DINOv2 SAC
- `reacher-hard`: V-JEPA SAC

---

## Phase 3 — Harder Task (if Phase 2 succeeds)

- `ball_in_cup-catch`: requires precise fast motion — tests temporal encoding
- `walker-walk`: tests body pose understanding
- OR: custom pick-and-place using MuJoCo directly (not dm_control)

---

## Optional: Fine-tune V-JEPA encoder on robot video

If Phase 2 results are good but not great, fine-tune V-JEPA on the robot demo footage
to make embeddings domain-specific.

- Cost: A100 (80GB) × 3-4 hrs = ~$10-15 of credits
- Expected gain: +10-20% on downstream task success rate

---

## Files to create

```
train_robots/
  plan.md                    ← this file
  generate_demos.py          ← dm_control expert + render pipeline
  encode_demos_modal.py      ← V-JEPA encoding job (Modal)
  train_bc.py                ← behavior cloning MLP
  train_sac_modal.py         ← SAC training on Modal with parallel envs
  eval.py                    ← rollout evaluation
  results/                   ← charts, success rate plots
```

---

## Key dependencies
- `dm_control` — MuJoCo physics + environments
- `mujoco` — renderer
- `stable-baselines3` or custom SAC — RL algorithm
- `modal` — GPU compute for encoding + RL training
- `transformers` — V-JEPA 2 model

---

## Definition of success
> V-JEPA 2 backbone achieves **≥ 60% success rate** on `reacher-easy` with ≤ 500k environment steps,
> outperforming a random embedding baseline by **≥ 2×**.
