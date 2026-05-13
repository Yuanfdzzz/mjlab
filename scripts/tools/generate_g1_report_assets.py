#!/usr/bin/env python3
"""Generate report figures for the G1 12DoF perceptive locomotion runs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.config import flat, open_stairs, random_stairs
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.torch import configure_torch_backends


TASK = "Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof"


def _read_latest_run() -> str:
  return Path("logs/latest_stride_run.txt").read_text().strip()


def _latest_checkpoint(run_name: str) -> Path:
  candidates = sorted(
    Path("logs/rsl_rl/g1_velocity").glob(f"*{run_name}/model_*.pt"),
    key=lambda p: p.stat().st_mtime,
  )
  if not candidates:
    raise FileNotFoundError(f"No checkpoints found for run {run_name!r}")
  return candidates[-1]


def _parse_training_log(log_path: Path) -> list[dict[str, float]]:
  records: list[dict[str, float]] = []
  current: dict[str, float] | None = None
  patterns = {
    "mean_reward": re.compile(r"Mean reward:\s*([-+0-9.eE]+)"),
    "episode_length": re.compile(r"Mean episode length:\s*([-+0-9.eE]+)"),
    "peak_foot_height_m": re.compile(r"Metrics/peak_height_mean:\s*([-+0-9.eE]+)"),
  }
  iter_re = re.compile(r"Learning iteration\s+(\d+)/")

  for line in log_path.read_text(errors="ignore").splitlines():
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    match = iter_re.search(line)
    if match:
      if current and len(current) > 1:
        records.append(current)
      current = {"iteration": float(match.group(1))}
      continue
    if current is None:
      continue
    for key, pattern in patterns.items():
      match = pattern.search(line)
      if match:
        current[key] = float(match.group(1))

  if current and len(current) > 1:
    records.append(current)
  return [r for r in records if {"mean_reward", "episode_length", "peak_foot_height_m"} <= r.keys()]


def _write_metrics_csv(records: Iterable[dict[str, float]], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  rows = list(records)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=[
        "iteration",
        "mean_reward",
        "episode_length",
        "peak_foot_height_m",
        "peak_foot_height_cm",
      ],
    )
    writer.writeheader()
    for row in rows:
      out = dict(row)
      out["peak_foot_height_cm"] = row["peak_foot_height_m"] * 100.0
      writer.writerow(out)


def _plot_training_curves(records: list[dict[str, float]], out_dir: Path) -> None:
  if not records:
    raise RuntimeError("No complete metric records found in the training log")

  x = np.array([r["iteration"] for r in records])
  reward = np.array([r["mean_reward"] for r in records])
  length = np.array([r["episode_length"] for r in records])
  peak_cm = np.array([r["peak_foot_height_m"] * 100.0 for r in records])

  out_dir.mkdir(parents=True, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")

  fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
  series = [
    (reward, "Average Return", "mean reward", "#2563eb"),
    (length, "Episode Length", "steps", "#059669"),
    (peak_cm, "Peak Foot Height", "cm", "#dc2626"),
  ]
  for ax, (y, title, ylabel, color) in zip(axes, series, strict=True):
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.scatter(x[-1], y[-1], color=color, s=32, zorder=3)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.annotate(
      f"{y[-1]:.2f}",
      xy=(x[-1], y[-1]),
      xytext=(8, 0),
      textcoords="offset points",
      va="center",
      fontsize=9,
      color=color,
    )
  axes[-1].set_xlabel("PPO learning iteration")
  fig.suptitle("G1 12DoF Perceptive Walking Training Metrics", fontsize=15, fontweight="bold")
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out_dir / "training_curves.png", dpi=180)
  fig.savefig(out_dir / "training_curves.svg")
  plt.close(fig)

  def rolling_mean(values: np.ndarray, window: int = 201) -> np.ndarray:
    if len(values) < window:
      return values
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

  # A robust view for reports: keep raw samples faintly visible but limit the y-axis
  # to central quantiles so single catastrophic rollouts do not hide the learning trend.
  fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
  for ax, (y, title, ylabel, color) in zip(axes, series, strict=True):
    lo, hi = np.quantile(y, [0.01, 0.99])
    margin = max((hi - lo) * 0.12, 1e-6)
    ax.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
    ax.plot(x, rolling_mean(y), color=color, linewidth=2.2)
    ax.scatter(x[-1], y[-1], color=color, s=32, zorder=3)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_title(f"{title} (Smoothed)", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.annotate(
      f"{y[-1]:.2f}",
      xy=(x[-1], y[-1]),
      xytext=(8, 0),
      textcoords="offset points",
      va="center",
      fontsize=9,
      color=color,
    )
  axes[-1].set_xlabel("PPO learning iteration")
  fig.suptitle(
    "G1 12DoF Training Metrics - Robust Smoothed View",
    fontsize=15,
    fontweight="bold",
  )
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out_dir / "training_curves_smoothed.png", dpi=180)
  fig.savefig(out_dir / "training_curves_smoothed.svg")
  plt.close(fig)


def _terrain_cfg(kind: str) -> TerrainGeneratorCfg:
  if kind == "flat":
    sub_terrains = {"flat": flat(proportion=1.0)}
  elif kind == "mixed":
    sub_terrains = {
      "low_up_stairs": open_stairs(
        proportion=0.25,
        step_height_range=(0.020, 0.080),
        step_width_range=(0.70, 1.00),
      ),
      "low_down_stairs": open_stairs(
        proportion=0.20,
        step_height_range=(0.020, 0.080),
        step_width_range=(0.70, 1.00),
        inverted=True,
      ),
      "mid_up_stairs": open_stairs(
        proportion=0.25,
        step_height_range=(0.085, 0.135),
        step_width_range=(0.68, 0.95),
      ),
      "mid_down_stairs": open_stairs(
        proportion=0.20,
        step_height_range=(0.060, 0.115),
        step_width_range=(0.68, 0.95),
        inverted=True,
      ),
      "uneven_up_stairs": random_stairs(
        proportion=0.10,
        step_width=0.76,
        step_height_range=(0.035, 0.120),
      ),
    }
  elif kind == "upstairs":
    sub_terrains = {
      "upstairs_13_5cm": open_stairs(
        proportion=1.0,
        step_height_range=(0.135, 0.135),
        step_width_range=(0.75, 0.75),
      )
    }
  elif kind == "downstairs":
    sub_terrains = {
      "downstairs_11_5cm": open_stairs(
        proportion=1.0,
        step_height_range=(0.115, 0.115),
        step_width_range=(0.75, 0.75),
        inverted=True,
      )
    }
  else:
    raise ValueError(f"Unknown terrain kind: {kind}")

  return TerrainGeneratorCfg(
    seed=17,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=1 if kind != "mixed" else 3,
    num_cols=1 if kind != "mixed" else 4,
    curriculum=False,
    difficulty_range=(1.0, 1.0),
    sub_terrains=sub_terrains,
    add_lights=True,
  )


def _make_env(kind: str, width: int, height: int, device: str) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _terrain_cfg(kind)
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.terminations = {}
  cfg.viewer.width = width
  cfg.viewer.height = height
  cfg.viewer.distance = 3.2
  cfg.viewer.elevation = -7.0
  cfg.viewer.azimuth = 90.0
  cfg.viewer.max_extra_envs = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env.update_visualizers = lambda visualizer: None
  return env


def _load_policy(env: ManagerBasedRlEnv, checkpoint: Path, device: str):
  agent_cfg = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return wrapped, runner.get_inference_policy(device=device)


def _foot_heights_cm(env: ManagerBasedRlEnv) -> tuple[float, float]:
  try:
    heights = env.scene["foot_height_scan"].data.heights[0].detach().cpu().numpy()
    return float(heights[0] * 100.0), float(heights[1] * 100.0)
  except Exception:
    return float("nan"), float("nan")


def _root_height_cm(env: ManagerBasedRlEnv) -> float:
  try:
    z = env.scene["robot"].data.root_link_pos_w[0, 2].detach().cpu().item()
    return float(z * 100.0)
  except Exception:
    return float("nan")


def _annotate(img: Image.Image, text: str) -> Image.Image:
  draw = ImageDraw.Draw(img)
  margin = 16
  try:
    bbox = draw.multiline_textbbox((0, 0), text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
  except Exception:
    w, h = (520, 42)
  draw.rectangle((margin - 8, margin - 8, margin + w + 8, margin + h + 8), fill=(255, 255, 255, 220))
  draw.multiline_text((margin, margin), text, fill=(20, 20, 20))
  return img


def _contact_sheet(paths: list[Path], title: str, out_path: Path) -> None:
  images = [Image.open(p).convert("RGB") for p in paths]
  thumb_w = 420
  thumbs = []
  for img in images:
    h = int(img.height * thumb_w / img.width)
    thumbs.append(img.resize((thumb_w, h), Image.Resampling.LANCZOS))
  pad = 14
  label_h = 34
  cols = len(thumbs)
  sheet_w = cols * thumb_w + (cols + 1) * pad
  sheet_h = max(t.height for t in thumbs) + 2 * pad + label_h + 46
  sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
  draw = ImageDraw.Draw(sheet)
  draw.text((pad, pad), title, fill=(20, 20, 20))
  y = pad + 46
  for i, img in enumerate(thumbs):
    x = pad + i * (thumb_w + pad)
    sheet.paste(img, (x, y))
    draw.text((x, y + img.height + 6), f"Frame {i + 1}", fill=(40, 40, 40))
  sheet.save(out_path)


def _capture_sequence(
  kind: str,
  title: str,
  checkpoint: Path,
  out_dir: Path,
  steps: list[int],
  total_steps: int,
  device: str,
) -> None:
  seq_dir = out_dir / kind
  seq_dir.mkdir(parents=True, exist_ok=True)
  env = _make_env(kind, width=1600, height=900, device=device)
  wrapped, policy = _load_policy(env, checkpoint, device=device)
  obs = wrapped.get_observations()
  saved: list[Path] = []
  step_set = set(steps)
  try:
    for step in range(total_steps + 1):
      if step in step_set:
        frame = env.render()
        if frame is not None:
          img = Image.fromarray(np.asarray(frame).astype(np.uint8))
          left_h, right_h = _foot_heights_cm(env)
          root_h = _root_height_cm(env)
          text = (
            f"{title}\n"
            f"t={step * env.step_dt:.2f}s  left foot={left_h:.1f}cm  "
            f"right foot={right_h:.1f}cm  pelvis={root_h:.1f}cm"
          )
          img = _annotate(img, text)
          path = seq_dir / f"{kind}_{step:04d}.png"
          img.save(path)
          saved.append(path)
      with torch.no_grad():
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
  finally:
    wrapped.close()
  _contact_sheet(saved, title, out_dir / f"{kind}_contact_sheet.png")


def generate(args: argparse.Namespace) -> None:
  configure_torch_backends()
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  run_name = args.run_name or _read_latest_run()
  log_path = Path("logs") / f"{run_name}.stdout.log"
  checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run_name)

  records = _parse_training_log(log_path)
  _write_metrics_csv(records, out_dir / "training_metrics.csv")
  _plot_training_curves(records, out_dir)

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  sequences = [
    ("flat", "Alternating Gait on Flat Ground", [20, 32, 44, 56, 68, 80], 90),
    ("mixed", "Random Discontinuous Terrain Walk", [30, 50, 70, 90, 110, 130], 140),
    ("upstairs", "Upstairs Sequence: 13.5cm Steps", [20, 38, 56, 74, 92, 110], 120),
    ("downstairs", "Downstairs Sequence: 11.5cm Steps", [20, 38, 56, 74, 92, 110], 120),
  ]
  for kind, title, steps, total_steps in sequences:
    _capture_sequence(kind, title, checkpoint, out_dir, steps, total_steps, device)

  summary = out_dir / "README.md"
  summary.write_text(
    "\n".join(
      [
        "# G1 Report Assets",
        "",
        f"Run: `{run_name}`",
        f"Checkpoint: `{checkpoint}`",
        "",
        "Generated files:",
        "- `training_curves.png` / `training_curves.svg`: average return, episode length, and peak foot height vs iteration.",
        "- `training_curves_smoothed.png` / `training_curves_smoothed.svg`: report-friendly robust view with a rolling average.",
        "- `training_metrics.csv`: parsed scalar data used for the curves.",
        "- `flat_contact_sheet.png`: alternating gait on flat ground.",
        "- `mixed_contact_sheet.png`: random discontinuous terrain walk.",
        "- `upstairs_contact_sheet.png`: upstairs sequence.",
        "- `downstairs_contact_sheet.png`: downstairs sequence.",
        "- Subdirectories contain individual annotated PNG frames.",
        "",
      ]
    ),
    encoding="utf-8",
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-name", default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--output-dir", default="report_assets/g1_perceptive_walk_20260513")
  parser.add_argument("--device", default=None)
  args = parser.parse_args()
  generate(args)


if __name__ == "__main__":
  main()
