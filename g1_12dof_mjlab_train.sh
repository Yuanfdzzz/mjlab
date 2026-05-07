#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${ROOT_DIR}/src/mjlab" ]; then
  DEFAULT_MJLAB_DIR="${ROOT_DIR}"
else
  DEFAULT_MJLAB_DIR="${ROOT_DIR}/mjlab"
fi
MJLAB_DIR="${MJLAB_DIR:-${DEFAULT_MJLAB_DIR}}"
TASK="${TASK:-Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_velocity}"
LOGGER="${LOGGER:-tensorboard}"
UPLOAD_MODEL="${UPLOAD_MODEL:-False}"
NUM_ENVS="${NUM_ENVS:-512}"
ITERS="${ITERS:-12000}"
RUN_NAME="${RUN_NAME:-g1_12dof_easy_disc_$(date +%Y%m%d_%H%M%S)}"
VIEWER="${VIEWER:-viser}"
TB_PORT="${TB_PORT:-6006}"
PLAY_PORT_HINT="${PLAY_PORT_HINT:-8080}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") check
  $(basename "$0") smoke [extra train args...]
  $(basename "$0") train [extra train args...]
  $(basename "$0") train-bg [extra train args...]
  $(basename "$0") resume [extra train args...]
  $(basename "$0") play [checkpoint.pt]
  $(basename "$0") tensorboard
  $(basename "$0") latest
  $(basename "$0") status

Defaults:
  MJLAB_DIR=${MJLAB_DIR}
  TASK=${TASK}
  NUM_ENVS=${NUM_ENVS}
  ITERS=${ITERS}

Common overrides:
  NUM_ENVS=1024 ITERS=20000 RUN_NAME=my_run $(basename "$0") train-bg
  TASK=Mjlab-Velocity-Discontinuous-Unitree-G1-12Dof $(basename "$0") smoke
  LOAD_RUN='.*easy_disc.*' LOAD_CHECKPOINT='model_.*.pt' $(basename "$0") resume
  CKPT=/path/to/model_5250.pt $(basename "$0") play

Migration checklist on a new machine:
  1. Keep mjlab and rsl_rl as sibling directories under the same workspace.
  2. Run ./setup_mjlab.sh if the mjlab virtualenv is not created yet.
  3. Run this script's check command before long training.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_mjlab() {
  [ -d "${MJLAB_DIR}" ] || die "MJLAB_DIR does not exist: ${MJLAB_DIR}"
  [ -x "${MJLAB_DIR}/.venv/bin/python" ] || die "missing ${MJLAB_DIR}/.venv/bin/python; run ./setup_mjlab.sh first"
  [ -x "${MJLAB_DIR}/.venv/bin/train" ] || die "missing ${MJLAB_DIR}/.venv/bin/train; mjlab env is incomplete"
}

latest_checkpoint() {
  local log_root="${MJLAB_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
  [ -d "${log_root}" ] || return 1
  find "${log_root}" -type f -name 'model_*.pt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

cmd_check() {
  require_mjlab
  cd "${MJLAB_DIR}"

  printf 'Workspace: %s\n' "${ROOT_DIR}"
  printf 'mjlab:     %s\n' "${MJLAB_DIR}"
  printf 'task:      %s\n' "${TASK}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
  fi

  CHECK_TASK="${TASK}" CHECK_ENVS="${CHECK_ENVS:-8}" CHECK_STEPS="${CHECK_STEPS:-4}" .venv/bin/python - <<'PY'
import os
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = os.environ["CHECK_TASK"]
cfg = load_env_cfg(task)
cfg.scene.num_envs = int(os.environ["CHECK_ENVS"])
cfg.observations["actor"].enable_corruption = False
cfg.events.pop("push_robot", None)

terrain = cfg.scene.terrain
if terrain is not None and terrain.terrain_generator is not None:
  gen = terrain.terrain_generator
  print("terrain rows/cols:", gen.num_rows, gen.num_cols)
  print("terrain names:", ", ".join(gen.sub_terrains.keys()))
  print("max init level:", terrain.max_init_terrain_level)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
env = ManagerBasedRlEnv(cfg=cfg, device=device)
obs, _ = env.reset()

def assert_finite(name, tensor):
  finite = torch.isfinite(tensor)
  if not bool(finite.all()):
    bad = int((~finite).sum().detach().cpu())
    raise RuntimeError(f"{name} has {bad} non-finite values")

for group, tensor in obs.items():
  print("obs", group, tuple(tensor.shape))
  assert_finite(f"obs/{group}", tensor)

actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=device)
for step in range(int(os.environ["CHECK_STEPS"])):
  obs, rew, terminated, truncated, _ = env.step(actions)
  assert_finite(f"reward/{step}", rew)
  for group, tensor in obs.items():
    assert_finite(f"obs/{group}/{step}", tensor)
  done = int((terminated | truncated).sum().detach().cpu())
  print(f"step {step + 1}: reward_mean={float(rew.mean().detach().cpu()):.4f}, done={done}")

print("action dim:", env.action_manager.total_action_dim)
env.close()
print("check OK")
PY
}

cmd_smoke() {
  require_mjlab
  cd "${MJLAB_DIR}"
  local smoke_name="${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
  .venv/bin/train "${TASK}" \
    --env.scene.num-envs="${SMOKE_ENVS:-64}" \
    --agent.max-iterations="${SMOKE_ITERS:-2}" \
    --agent.logger="${LOGGER}" \
    --agent.upload-model=False \
    --agent.run-name="${smoke_name}" \
    "$@"
}

cmd_train() {
  require_mjlab
  cd "${MJLAB_DIR}"
  .venv/bin/train "${TASK}" \
    --env.scene.num-envs="${NUM_ENVS}" \
    --agent.max-iterations="${ITERS}" \
    --agent.logger="${LOGGER}" \
    --agent.upload-model="${UPLOAD_MODEL}" \
    --agent.run-name="${RUN_NAME}" \
    "$@"
}

cmd_train_bg() {
  require_mjlab
  cd "${MJLAB_DIR}"
  mkdir -p logs
  local stdout_log="logs/${RUN_NAME}.stdout.log"
  local pid_file="logs/${RUN_NAME}.pid"

  setsid env PYTHONUNBUFFERED=1 .venv/bin/train "${TASK}" \
    --env.scene.num-envs="${NUM_ENVS}" \
    --agent.max-iterations="${ITERS}" \
    --agent.logger="${LOGGER}" \
    --agent.upload-model="${UPLOAD_MODEL}" \
    --agent.run-name="${RUN_NAME}" \
    "$@" \
    > "${stdout_log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  printf 'Started background training\n'
  printf 'PID: %s\n' "${pid}"
  printf 'stdout: %s/%s\n' "${MJLAB_DIR}" "${stdout_log}"
  printf 'pid file: %s/%s\n' "${MJLAB_DIR}" "${pid_file}"
}

cmd_resume() {
  require_mjlab
  cd "${MJLAB_DIR}"
  local load_run="${LOAD_RUN:-.*}"
  local load_checkpoint="${LOAD_CHECKPOINT:-model_.*.pt}"
  .venv/bin/train "${TASK}" \
    --env.scene.num-envs="${NUM_ENVS}" \
    --agent.max-iterations="${ITERS}" \
    --agent.logger="${LOGGER}" \
    --agent.upload-model="${UPLOAD_MODEL}" \
    --agent.run-name="${RUN_NAME}" \
    --agent.resume=True \
    --agent.load-run="${load_run}" \
    --agent.load-checkpoint="${load_checkpoint}" \
    "$@"
}

cmd_play() {
  require_mjlab
  cd "${MJLAB_DIR}"
  local ckpt="${1:-${CKPT:-}}"
  if [ -z "${ckpt}" ]; then
    ckpt="$(latest_checkpoint)" || die "no checkpoint found; pass one as an argument or set CKPT"
  fi
  printf 'Playing checkpoint: %s\n' "${ckpt}"
  printf 'If using viser, open http://localhost:%s\n' "${PLAY_PORT_HINT}"
  .venv/bin/play "${TASK}" \
    --checkpoint-file="${ckpt}" \
    --num-envs="${PLAY_ENVS:-1}" \
    --viewer="${VIEWER}" \
    --no-terminations="${PLAY_NO_TERMINATIONS:-False}"
}

cmd_tensorboard() {
  require_mjlab
  cd "${MJLAB_DIR}"
  printf 'Open http://localhost:%s\n' "${TB_PORT}"
  .venv/bin/tensorboard --logdir logs/rsl_rl --host 0.0.0.0 --port "${TB_PORT}"
}

cmd_latest() {
  require_mjlab
  latest_checkpoint || die "no checkpoint found"
}

cmd_status() {
  require_mjlab
  cd "${MJLAB_DIR}"
  printf 'Training processes:\n'
  pgrep -af '\.venv/bin/train|train Mjlab-Velocity' || true
  printf '\nGPU:\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
  else
    printf 'nvidia-smi not found\n'
  fi
  printf '\nLatest checkpoint:\n'
  latest_checkpoint || true
  printf '\nRecent run directories:\n'
  find "logs/rsl_rl/${EXPERIMENT_NAME}" -maxdepth 1 -mindepth 1 -type d -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null \
    | sort \
    | tail -n 5
}

main() {
  local cmd="${1:-help}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  case "${cmd}" in
    check) cmd_check "$@" ;;
    smoke) cmd_smoke "$@" ;;
    train) cmd_train "$@" ;;
    train-bg) cmd_train_bg "$@" ;;
    resume) cmd_resume "$@" ;;
    play) cmd_play "$@" ;;
    tensorboard) cmd_tensorboard "$@" ;;
    latest) cmd_latest "$@" ;;
    status) cmd_status "$@" ;;
    help|-h|--help) usage ;;
    *) usage; die "unknown command: ${cmd}" ;;
  esac
}

main "$@"
