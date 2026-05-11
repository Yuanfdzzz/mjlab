# G1 12DoF Current Best - Stairs Straight 170500

Saved from:

`logs/rsl_rl/g1_velocity/2026-05-11_16-07-12_g1_12dof_stage3_stairs_straight_continue_20260511_160706/model_170500.pt`

Reason for saving:

- User-confirmed best visual gait as of 2026-05-11.
- Natural-looking walking compared with previous runs.
- Balanced left/right swing timing and foot height in the latest metrics.
- Good straight-walking behavior on the current low up/down stair curriculum.

Files:

- `model_170500.pt`: checkpoint to resume or visualize.
- `model_170500.onnx`: exported policy.
- `agent.yaml`: runner/training config from this saved run.
- `env.yaml`: environment config from this saved run.
- `mjlab.diff`: code diff captured by the training run.

Recommended resume source:

`LOAD_RUN='.*g1_12dof_stage3_stairs_straight_continue_20260511_160706' LOAD_CHECKPOINT='model_170500.pt'`

Next training intent:

Keep this gait while gradually raising stair height. Do not resume from later checkpoints
unless they are visually checked.
