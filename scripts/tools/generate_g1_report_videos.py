#!/usr/bin/env python3
"""Generate short MP4 clips for the G1 perceptive walking presentation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw

from generate_g1_report_assets import (
  TASK,
  _foot_heights_cm,
  _latest_checkpoint,
  _load_policy,
  _make_env,
  _read_latest_run,
  _root_height_cm,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.config import box_random_grid
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.torch import configure_torch_backends


def _annotate_frame(
  frame: np.ndarray,
  title: str,
  t: float,
  env,
  camera_label: str,
) -> np.ndarray:
  img = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
  left_h, right_h = _foot_heights_cm(env)
  root_h = _root_height_cm(env)
  text = (
    f"{title}\n"
    f"t={t:4.2f}s   left foot={left_h:4.1f}cm   "
    f"right foot={right_h:4.1f}cm   pelvis={root_h:5.1f}cm   {camera_label}"
  )
  draw = ImageDraw.Draw(img)
  pad = 14
  bbox = draw.multiline_textbbox((0, 0), text)
  w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
  draw.rectangle((pad - 8, pad - 8, pad + w + 8, pad + h + 8), fill=(255, 255, 255))
  draw.multiline_text((pad, pad), text, fill=(20, 20, 20))
  return np.asarray(img)


def _mixed_showcase_terrain_cfg() -> TerrainGeneratorCfg:
  """A moderate discontinuous terrain for presentation video.

  The training-time random mixed grid is useful for stress testing, but it can sample
  very awkward patches for a short report clip. This single-patch grid keeps the
  terrain discontinuous while avoiding an immediate worst-case spawn.
  """

  return TerrainGeneratorCfg(
    seed=29,
    size=(10.0, 10.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    difficulty_range=(0.75, 0.75),
    sub_terrains={
      "moderate_random_blocks": box_random_grid(
        proportion=1.0,
        grid_width=0.78,
        grid_height_range=(0.015, 0.080),
        platform_width=1.45,
      )
    },
    add_lights=True,
  )


def _make_video_env(
  kind: str,
  width: int,
  height: int,
  device: str,
  camera_azimuth: float,
  camera_elevation: float,
  camera_distance: float,
) -> ManagerBasedRlEnv:
  if kind != "mixed_showcase":
    env = _make_env(kind, width=width, height=height, device=device)
    _set_camera(
      env,
      azimuth=camera_azimuth,
      elevation=camera_elevation,
      distance=camera_distance,
    )
    return env

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _mixed_showcase_terrain_cfg()
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.events.pop("randomize_terrain", None)
  cfg.terminations = {}
  cfg.viewer.width = width
  cfg.viewer.height = height
  cfg.viewer.distance = camera_distance
  cfg.viewer.elevation = camera_elevation
  cfg.viewer.azimuth = camera_azimuth
  cfg.viewer.max_extra_envs = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env.update_visualizers = lambda visualizer: None
  return env


def _set_camera(
  env,
  *,
  azimuth: float,
  elevation: float,
  distance: float,
) -> None:
  env.cfg.viewer.azimuth = float(azimuth)
  env.cfg.viewer.elevation = float(elevation)
  env.cfg.viewer.distance = float(distance)
  renderer = getattr(env, "_offline_renderer", None)
  cam = getattr(renderer, "_cam", None)
  if cam is not None:
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)


def _capture_video(
  kind: str,
  title: str,
  checkpoint: Path,
  out_path: Path,
  device: str,
  seconds: float = 12.0,
  fps: int = 30,
  width: int = 1280,
  height: int = 720,
  camera_azimuth: float = 115.0,
  camera_sweep: float = 0.0,
  camera_elevation: float = -8.0,
  camera_distance: float = 3.4,
) -> None:
  env = _make_video_env(
    kind,
    width=width,
    height=height,
    device=device,
    camera_azimuth=camera_azimuth,
    camera_elevation=camera_elevation,
    camera_distance=camera_distance,
  )
  wrapped, policy = _load_policy(env, checkpoint, device=device)
  obs = wrapped.get_observations()
  frames = []
  sim_dt = float(env.step_dt)
  render_interval = max(1, round((1.0 / fps) / sim_dt))
  output_fps = 1.0 / (render_interval * sim_dt)
  total_steps = int(seconds / sim_dt)
  try:
    for step in range(total_steps):
      t = step * sim_dt
      azimuth = camera_azimuth + camera_sweep * min(t / max(seconds, 1e-6), 1.0)
      camera_label = f"camera az={azimuth:5.1f}deg"
      if step % render_interval == 0:
        _set_camera(
          env,
          azimuth=azimuth,
          elevation=camera_elevation,
          distance=camera_distance,
        )
        frame = env.render()
        if frame is not None:
          frames.append(_annotate_frame(frame, title, t, env, camera_label))
      with torch.no_grad():
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
  finally:
    wrapped.close()

  out_path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(out_path), frames, fps=output_fps)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-name", default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--output-dir", default="report_assets/g1_perceptive_walk_20260513/videos")
  parser.add_argument("--device", default=None)
  parser.add_argument("--seconds", type=float, default=0.0)
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument(
    "--only",
    choices=["all", "flat", "upstairs", "downstairs", "mixed"],
    default="all",
  )
  args = parser.parse_args()

  configure_torch_backends()
  run_name = args.run_name or _read_latest_run()
  checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run_name)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  out_dir = Path(args.output_dir)

  jobs = [
    ("flat", "Flat Ground Alternating Gait", "flat_walk.mp4", 12.0, 125.0, 55.0),
    ("upstairs", "Upstairs: 13.5cm Steps", "upstairs_13_5cm.mp4", 12.0, 120.0, 55.0),
    ("downstairs", "Downstairs: 11.5cm Steps", "downstairs_11_5cm.mp4", 12.0, 120.0, 55.0),
    (
      "mixed_showcase",
      "Moderate Discontinuous Terrain",
      "mixed_discontinuous.mp4",
      12.0,
      45.0,
      180.0,
    ),
  ]
  for kind, title, filename, default_seconds, camera_azimuth, camera_sweep in jobs:
    if args.only != "all" and args.only not in kind:
      continue
    duration = args.seconds if args.seconds > 0 else default_seconds
    print(f"[video] {title} -> {out_dir / filename}")
    _capture_video(
      kind=kind,
      title=title,
      checkpoint=checkpoint,
      out_path=out_dir / filename,
      device=device,
      seconds=duration,
      fps=args.fps,
      camera_azimuth=camera_azimuth,
      camera_sweep=camera_sweep,
    )

  (out_dir / "README.md").write_text(
    "\n".join(
      [
        "# G1 Report Videos",
        "",
        f"Task: `{TASK}`",
        f"Run: `{run_name}`",
        f"Checkpoint: `{checkpoint}`",
        "",
        "- `flat_walk.mp4`: flat-ground alternating gait.",
        "- `upstairs_13_5cm.mp4`: upstairs sequence with 13.5cm steps.",
        "- `downstairs_11_5cm.mp4`: downstairs sequence with 11.5cm steps.",
        "- `mixed_discontinuous.mp4`: moderate discontinuous terrain with a sweeping camera.",
        "",
      ]
    ),
    encoding="utf-8",
  )


if __name__ == "__main__":
  main()
