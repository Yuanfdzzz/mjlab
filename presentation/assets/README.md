# G1 Report Assets

Run: `g1_12dof_stage6_135cm_to_15cm_prep_20260512_121623`
Checkpoint: `logs/rsl_rl/g1_velocity/2026-05-12_12-16-28_g1_12dof_stage6_135cm_to_15cm_prep_20260512_121623/model_314497.pt`

Generated files:
- `training_curves.png` / `training_curves.svg`: average return, episode length, and peak foot height vs iteration.
- `training_curves_smoothed.png` / `training_curves_smoothed.svg`: report-friendly robust view with a rolling average.
- `training_metrics.csv`: parsed scalar data used for the curves.
- `flat_contact_sheet.png`: alternating gait on flat ground.
- `mixed_contact_sheet.png`: random discontinuous terrain walk.
- `upstairs_contact_sheet.png`: upstairs sequence.
- `downstairs_contact_sheet.png`: downstairs sequence.
- `videos/`: 12-second MP4 clips for flat walking, upstairs, downstairs, and moderate discontinuous terrain.
- Subdirectories contain individual annotated PNG frames.
