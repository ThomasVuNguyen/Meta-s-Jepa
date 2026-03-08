# Experiment 9: Stabilized Reward Head (Walker-Walk)

## Objective
In Experiment 8, we observed a catastrophic collapse in performance during Round 2 (win rate dropped from 80% to 40%). We hypothesized this was due to the reward head overfitting to the narrow on-policy states visited during Round 1. 
The goal of Experiment 9 was to stabilize the planning by **freezing the reward head** after its initial training in Round 1, preventing it from incorrectly labelling out-of-distribution terminal states during subsequent rounds. Fine-tuning would only be applied to the dynamics model in Round 2 and Round 3.

## Method
- **Base encoder:** V-JEPA 2 (frozen)
- **Dynamics Warm-Start:** Dynamics MLP trained on 20k offline random transitions from Exp 6.
- **Reward Head:** MLP reward predictor. Trained at Round 0, retrained at Round 1, and then **frozen** for Round 2 and Round 3.
- **Dyna Loop:** 
  - **Round 0:** Baseline evaluation (Exp6 Dynamics + Initial Reward Head).
  - **Round 1:** Collect 60 on-policy rollouts (6,000 steps). Fine-tune dynamics, retraining reward head.
  - **Round 2:** Collect 60 on-policy rollouts. Fine-tune dynamics, **freeze** reward head.
  - **Round 3:** Collect 60 on-policy rollouts... (Job timed out during collection).

## Results

*Note: The script experienced a 1-hour timeout during Round 3 collection on Modal, but Rounds 0-2 completed successfully.*

| Round | MPC Reward | Random Reward | Win Rate vs Random | Dynamics Val Loss | Reward Val Loss |
|-------|------------|---------------|--------------------|-------------------|-----------------|
| R0    | 6.99       | 4.71          | 80%                | 0.0377 (base)     | 0.1355          |
| R1    | 6.84       | 6.51          | 50%                | 0.0316            | 0.1175          |
| R2    | 4.77       | 5.01          | 60%                | 0.0223            | 0.1175 (frozen) |

## Analysis
1. **Initial Performance (R0):** The baseline MPC agent with the initial reward head performed strongly, achieving an 80% win rate and a score of 6.99.
2. **First Fine-tuning (R1):** Retraining the reward head and fine-tuning dynamics resulted in a drop in win rate to 50%, though the average MPC reward remained stable (6.84). 
3. **Frozen Reward Head (R2):** In Round 2, freezing the reward head prevented the severe 40% collapse seen in Exp 8, achieving a 60% win rate. However, the absolute MPC reward dropped to 4.77.
4. **Insights:** Freezing the reward head is helpful for avoiding severe overestimation of bad states, but it does not completely prevent degradation in performance. This suggests that the dynamics model itself is also suffering from severe compounding errors when fine-tuned on the narrow on-policy distributions. The dynamics validation loss decreases (0.037 -> 0.022), indicating that the model is fitting the on-policy data very well, but likely losing global validity required for the CEM search.

## Next Steps
To truly stabilize the Dyna loop, we must prevent the dynamics model from catastrophically forgetting the global state-space. Potential solutions:
- **Replay Buffers:** Maintain a mixture of initial random rollouts and recent on-policy rollouts during standard dynamics fine-tuning.
- **Target Networks:** Use exponentially moving averages (EMA) for the dynamics weights to prevent rapid overwriting.
- **KL Penalties / Ensemble Uncertainty:** Ensure that the CEM planner is discouraged from steering into regions where the dynamics model has lost confidence due to localized fine-tuning.
