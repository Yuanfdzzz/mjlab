# G1 12DoF stride-extend baseline

This directory stores the current baseline requested on 2026-05-08.

Source run:

```text
logs/rsl_rl/g1_velocity/2026-05-08_11-15-24_g1_12dof_stride_extend_stairs_20260508_111519
```

Primary checkpoint:

```text
model_109499.pt
```

Stable resume run copied for training scripts:

```text
logs/rsl_rl/g1_velocity/BASE_g1_12dof_stride_extend_109499_20260508
```

Use this checkpoint as the base for further gait work. It has improved alternating
foot contacts, higher foot clearance, and a larger sagittal step than earlier
versions, but still needs work for a more natural long stride and higher stairs.

