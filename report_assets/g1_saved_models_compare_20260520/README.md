# G1 Saved Model Comparison Clips

These clips were rendered from previously saved checkpoints in `saved_models/`.
Each clip is a 6-second flat-ground gait sample at 1280x720 and 25 fps.

Checkpoint clips:

- `g1_12dof_gaitfix_21748_20260507/videos/flat_walk.mp4`
- `g1_12dof_stride_extend_109499_20260508/videos/flat_walk.mp4`
- `g1_12dof_current_best_recover_walk_161500_20260511/videos/flat_walk.mp4`
- `g1_12dof_current_best_stairs_straight_170500_20260511/videos/flat_walk.mp4`
- `g1_12dof_current_best_12cm_ready_182499_20260511/videos/flat_walk.mp4`
- `g1_12dof_best_12cm_gait_248498_20260512/videos/flat_walk.mp4`

Preview:

- `saved_models_flat_compare_preview.png`

Generation command pattern:

```bash
.venv/bin/python scripts/tools/generate_g1_report_videos.py \
  --checkpoint saved_models/<model_dir>/model_<iter>.pt \
  --output-dir report_assets/g1_saved_models_compare_20260520/<model_dir>/videos \
  --only flat \
  --seconds 6
```
