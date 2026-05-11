# G1 12DoF Current Best - Recover Walk 161500

Saved from:

`logs/rsl_rl/g1_velocity/2026-05-11_12-48-36_g1_12dof_stage2_recover_walk_20260511_124831/model_161500.pt`

Reason for saving:

- Current best visual result as of 2026-05-11.
- Recovers walking after the failed `stage2_grounded_natural` run.
- No observed low-pelvis collapse in the short training window.
- More balanced left/right foot height than the previous right-foot-dominant model.

Files:

- `model_161500.pt`: checkpoint to resume or visualize.
- `model_161500.onnx`: exported policy at this run state.
- `agent.yaml`: runner/training config from the saved run.
- `env.yaml`: environment config from the saved run.
- `mjlab.diff`: code diff captured by the training run.

Recommended resume source:

`LOAD_RUN='.*g1_12dof_stage2_recover_walk_20260511_124831' LOAD_CHECKPOINT='model_161500.pt'`
