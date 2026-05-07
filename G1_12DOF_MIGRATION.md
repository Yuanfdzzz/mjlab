# Unitree G1 12DoF Easy Discontinuous Training

This branch contains a self-contained mjlab setup for Unitree G1 12DoF
perceptive walking on easier discontinuous terrain: flat ground, low stairs,
low random stairs, small-gap platforms, and a light box grid. The 12DoF MJCF
and meshes are vendored under `src/mjlab/asset_zoo/robots/unitree_g1_12dof/xmls`,
so a sibling `rsl_rl` checkout is no longer required for this asset.

## Quick Start

From the repository root:

```bash
./g1_12dof_mjlab_train.sh check
./g1_12dof_mjlab_train.sh smoke
NUM_ENVS=512 ITERS=12000 ./g1_12dof_mjlab_train.sh train-bg
```

Useful commands:

```bash
./g1_12dof_mjlab_train.sh status
./g1_12dof_mjlab_train.sh tensorboard
./g1_12dof_mjlab_train.sh latest
./g1_12dof_mjlab_train.sh play
```

The default task is:

```text
Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof
```

For harder terrain, override the task:

```bash
TASK=Mjlab-Velocity-Discontinuous-Unitree-G1-12Dof ./g1_12dof_mjlab_train.sh train-bg
```

## New Machine

From a bundle:

```bash
git clone /path/to/mjlab-g1-12dof-easy-discontinuous.bundle mjlab
cd mjlab
git switch g1-12dof-easy-discontinuous-migration
```

Create the environment:

```bash
uv sync --python 3.10 --extra cu128 --no-dev
```

Then run:

```bash
./g1_12dof_mjlab_train.sh check
./g1_12dof_mjlab_train.sh smoke
NUM_ENVS=512 ITERS=12000 ./g1_12dof_mjlab_train.sh train-bg
```

If GPU memory is lower than this machine's RTX 4060 8GB, start with
`NUM_ENVS=256`. If memory is larger, try `NUM_ENVS=1024`.

The script stores stdout logs in `logs/<run-name>.stdout.log`; mjlab/rsl_rl
stores checkpoints and TensorBoard files in `logs/rsl_rl/g1_velocity/`.
