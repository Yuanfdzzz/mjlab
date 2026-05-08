# G1 12DoF MJLab 训练说明

本文档记录本机当前 G1 12DoF 下肢模型的训练框架、环境参数、奖励函数改动和最终训练结果。对应工作目录为 `/home/robotics/mjlab`。

## 当前结论

最终使用的 run 为：

```bash
logs/rsl_rl/g1_velocity/2026-05-08_11-15-24_g1_12dof_stride_extend_stairs_20260508_111519
```

最终模型为：

```bash
logs/rsl_rl/g1_velocity/2026-05-08_11-15-24_g1_12dof_stride_extend_stairs_20260508_111519/model_109499.pt
```

该模型已固定保存为基准模型：

```bash
saved_models/g1_12dof_stride_extend_109499_20260508/model_109499.pt
```

并复制到 RSL-RL 可直接续训的稳定 run 目录：

```bash
logs/rsl_rl/g1_velocity/BASE_g1_12dof_stride_extend_109499_20260508/model_109499.pt
```

`g1_12dof_mjlab_train.sh` 现在默认会从这个 base checkpoint 续训。若要从零开始训练，需要显式设置：

```bash
RESUME_FROM_BASE=False ./g1_12dof_mjlab_train.sh train-bg
```

这一版已经完成训练。最后几轮的主要结果：

- `alternating_landing_rate` 约 `0.94`：左右脚落地大部分是交替的。
- `repeated_landing_rate` 约 `0.04`：重复同一只脚落地已经很低。
- `sagittal_passed_landing_rate` 约 `0.89`：落地脚大多会在前进方向越过另一只脚。
- `foot_x_separation_mean` 约 `0.113m`：前后脚距离比上一版 `0.065m` 明显变大，但仍小于目标 `0.16m`。
- `rear_foot_x_mean` 约 `-0.038m`：后脚已经能到身体后方，不再全挤在身体前面。
- `peak_height_mean` 约 `0.123m`：脚抬高比早期版本明显提升，但还没有完全达到目标 `0.16m`。
- `nan_detection` 为 `0`，最后阶段没有 NaN。

## 训练框架

当前任务入口：

```bash
TASK=Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof
```

主要组件：

- 仿真：MuJoCo / MuJoCo-Warp。
- 环境：MJLab `ManagerBasedRlEnv`。
- 算法：RSL-RL PPO 风格 actor-critic。
- 机器人：Unitree G1 12DoF lower-body only。
- 动作空间：12 维关节位置控制，`JointPositionActionCfg`。
- 动作 scale：`G1_12DOF_ACTION_SCALE = 0.25`。
- 观测：
  - actor 观测维度：`393`
  - critic 观测维度：`405`
  - actor 包含 base velocity、gravity projection、joint pos/vel、previous action、command、height scan。
  - critic 额外包含 foot height、foot air time、foot contact、foot contact forces。
- 策略网络：
  - actor MLP：`393 -> 512 -> 256 -> 128 -> 12`
  - critic MLP：`405 -> 512 -> 256 -> 128 -> 1`

训练辅助脚本：

```bash
./g1_12dof_mjlab_train.sh
```

常用命令：

```bash
# 环境检查
./g1_12dof_mjlab_train.sh check

# 后台训练
RUN_NAME=my_run NUM_ENVS=512 ITERS=30000 ./g1_12dof_mjlab_train.sh train-bg

# 从零开始后台训练
RESUME_FROM_BASE=False RUN_NAME=fresh_run ./g1_12dof_mjlab_train.sh train-bg

# 可视化指定 checkpoint
CKPT=/home/robotics/mjlab/logs/rsl_rl/g1_velocity/.../model_109499.pt \
PLAY_ENVS=1 VIEWER=viser PLAY_NO_TERMINATIONS=False \
./g1_12dof_mjlab_train.sh play

# TensorBoard
./g1_12dof_mjlab_train.sh tensorboard
```

可视化地址默认是：

```bash
http://localhost:8080
```

TensorBoard 地址默认是：

```bash
http://localhost:6006
```

## 环境参数

当前 G1 12DoF 训练配置在：

```bash
src/mjlab/tasks/velocity/config/g1/env_cfgs.py
```

使用的环境函数：

```python
unitree_g1_12dof_easy_discontinuous_env_cfg
```

### 地形课程

12DoF 专用地形函数：

```python
_g1_12dof_gait_terrains_cfg()
```

地形生成参数：

- terrain size：`8.0m x 8.0m`
- rows / cols：`6 x 3`
- curriculum：启用
- max init terrain level：`0`
- terrain border：`20.0m`

子地形比例：

- `flat`：`0.50`
- `gentle_up_slope`：`0.35`
  - slope range：`0.0 - 0.25`
  - platform width：`2.0m`
- `low_open_stairs`：`0.15`
  - step height range：`0.040 - 0.140m`
  - step width range：`0.65 - 0.95m`

说明：台阶高度已经从早期的 `0.015 - 0.055m`、`0.025 - 0.090m` 提高到当前的 `0.040 - 0.140m`。

### 速度命令

当前只训练直行，不训练横移和转向：

- `heading_command = False`
- `lin_vel_x = 0.45 - 0.90 m/s`
- `lin_vel_y = 0.0`
- `ang_vel_z = 0.0`
- `rel_standing_envs = 0.0`
- `rel_heading_envs = 0.0`
- `rel_forward_envs = 0.0`
- `init_velocity_prob = 0.0`
- `resampling_time_range = 4.0 - 7.0s`

命令课程：

- 起始：`lin_vel_x = 0.45 - 0.75 m/s`
- 后期：`lin_vel_x = 0.45 - 0.85 m/s`

play 模式下固定前进速度：

```python
lin_vel_x = 0.45
lin_vel_y = 0.0
ang_vel_z = 0.0
```

### 随机扰动和终止

为避免 12DoF 下肢模型早期被扰动打散：

- 移除了 `push_robot` 事件。
- 保留足底摩擦、编码器偏差、base mass/COM 等 startup randomization。
- observation `nan_policy = "sanitize"`。

新增终止项：

- `nan_detection`
- `pelvis_too_low`
  - pelvis 最低高度：`0.55m`

play 脚本中 `--no-terminations` 改为可配置：

```bash
PLAY_NO_TERMINATIONS=False
```

## 姿态先验

12DoF 下肢模型只包含腿部关节，所以 posture prior 主要放宽 hip/knee/ankle 的运动范围。

standing：

```python
{".*": 0.05}
```

walking：

```python
hip_pitch: 0.70
hip_roll: 0.25
hip_yaw: 0.20
knee: 0.85
ankle_pitch: 0.45
ankle_roll: 0.16
```

running：

```python
hip_pitch: 0.90
hip_roll: 0.30
hip_yaw: 0.25
knee: 1.00
ankle_pitch: 0.55
ankle_roll: 0.20
```

这些值的含义是：让髋俯仰、膝关节、踝俯仰有更大活动空间，避免模型只能僵硬地拖脚或小碎步。

## 奖励函数

当前 12DoF 任务共有 24 个 active reward terms。基础奖励来自 MJLab velocity task，后续增加了多项 gait shaping。

### 基础速度和姿态奖励

| 奖励项 | 权重 | 作用 |
| --- | ---: | --- |
| `track_linear_velocity` | `6.0` | 跟踪前进速度，当前 std 为 `0.22`。 |
| `track_angular_velocity` | `0.4` | 保持角速度接近命令；当前转向命令为 0。 |
| `upright` | `0.8` | 保持 pelvis 竖直。 |
| `pose` | `0.25` | 不要偏离默认姿态太夸张，但已经放宽腿部关节。 |
| `body_ang_vel` | `-0.05` | 抑制身体横滚/俯仰角速度。 |
| `angular_momentum` | `-0.02` | 抑制过大整体角动量。 |
| `dof_pos_limits` | `-1.0` | 避免关节撞限位。 |
| `action_rate_l2` | `-0.015` | 避免动作高频抖动。 |

### 抬脚、落脚和防滑

| 奖励项 | 权重 | 参数 | 作用 |
| --- | ---: | --- | --- |
| `air_time` | `0.6` | `threshold_min=0.08`, `threshold_max=0.46` | 鼓励有清晰摆动相。 |
| `foot_clearance` | `-0.20` | `target_height=0.16m` | 脚移动时高度接近目标，避免拖地。 |
| `foot_swing_height` | `-0.8` | `target_height=0.16m` | 落地时检查本次摆动峰值是否达到目标。 |
| `foot_slip` | `-0.35` | contact 时脚水平速度 | 抑制支撑脚打滑。 |
| `soft_landing` | `-2e-5` | landing force | 降低落地冲击。 |
| `self_collisions` | `-1.0` | pelvis self collision | 抑制明显自碰撞。 |

### 防单腿跳和防双脚腾空

这些奖励在：

```bash
src/mjlab/tasks/velocity/mdp/rewards.py
```

| 奖励项 | 权重 | 参数 | 作用 |
| --- | ---: | --- | --- |
| `alternating_foot_contacts` | `1.0` | contact sensor | 奖励左右脚交替 touchdown。 |
| `feet_air_time_limit` | `-3.0` | `max_air_time=0.50s` | 避免一条腿长时间抬着跳。 |
| `feet_air_time_symmetry` | `-1.0` | accumulated air time | 避免左右脚摆动时间长期不平衡。 |
| `no_flight_phase` | `-0.8` | both feet airborne | 避免变成跳跃 gait。 |

对应 TensorBoard 指标：

- `Metrics/alternating_landing_rate`
- `Metrics/repeated_landing_rate`
- `Metrics/max_foot_air_time`
- `Metrics/foot_air_time_imbalance`
- `Metrics/both_feet_airborne_rate`

### 前后脚交换和步幅奖励

这些是为了解决“左右脚虽然交替，但前后位置不换”“腿一直向前弯”“小碎步”的问题。

| 奖励项 | 权重 | 参数 | 作用 |
| --- | ---: | --- | --- |
| `sagittal_step_landing` | `2.2` | `pass_margin=0.10m` | touchdown 的脚需要在前进方向越过另一只脚。 |
| `sagittal_foot_order_switch` | `1.2` | `deadband=0.08m` | 奖励左右脚谁在前发生交换。 |
| `sagittal_foot_order_stall` | `-2.0` | `max_same_order_time=0.90s` | 惩罚同一只脚长期在前。 |
| `sagittal_foot_separation` | `-12.0` | `min_separation=0.16m` | 惩罚前后脚距离太近。 |
| `sagittal_stride_centering` | `-8.0` | `center_target=-0.02m`, `tolerance=0.04m` | 避免两只脚整体都挤在 pelvis 前方。 |
| `sagittal_front_rear_split` | `-10.0` | `front>=0.06m`, `rear<=-0.06m` | 要求一只脚在身体前方，另一只脚在身体后方。 |

对应 TensorBoard 指标：

- `Metrics/sagittal_passed_landing_rate`
- `Metrics/sagittal_order_switch_rate`
- `Metrics/sagittal_order_time_mean`
- `Metrics/foot_x_separation_mean`
- `Metrics/foot_x_separation_shortfall`
- `Metrics/foot_x_center_mean`
- `Metrics/foot_x_center_abs_error`
- `Metrics/front_foot_x_mean`
- `Metrics/rear_foot_x_mean`

## 训练历程摘要

几个关键阶段：

1. `g1_12dof_gaitfix_slow_20260507_212814`
   - checkpoint：`model_21748.pt`
   - 结果：能稳定前进，但出现一条腿长时间抬起、近似跳过去的问题。
   - 已备份：
     ```bash
     saved_models/g1_12dof_gaitfix_21748_20260507/model_21748.pt
     ```

2. `g1_12dof_alternating_stride_stairs_20260507_234146`
   - 从 `model_22500.pt` 接续。
   - 加入交替落脚、空中时间限制和更高台阶。
   - 结果：左右脚交替改善，但前后脚距离仍然偏小。

3. `g1_12dof_sagittal_swap_20260508_022738`
   - 从 `model_36500.pt` 接续。
   - 加入脚前后顺序交换奖励。
   - 结果：解决“长期一脚在前”的问题，但步态变成小碎步。

4. `g1_12dof_stride_extend_stairs_20260508_111519`
   - 从 `model_79500.pt` 接续。
   - 加入大步幅、前后脚分离、身体前后居中和更高台阶。
   - 最终模型：`model_109499.pt`
   - 结果：抬脚更高，后脚开始到身体后方，步幅增加，但仍未达到理想自然大步。

## 如何查看当前模型

启动最终模型：

```bash
CKPT=/home/robotics/mjlab/logs/rsl_rl/g1_velocity/2026-05-08_11-15-24_g1_12dof_stride_extend_stairs_20260508_111519/model_109499.pt \
PLAY_ENVS=1 VIEWER=viser PLAY_NO_TERMINATIONS=False \
./g1_12dof_mjlab_train.sh play
```

浏览器打开：

```bash
http://localhost:8080
```

停止可视化：

```bash
pkill -f '.venv/bin/play Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof'
```

## 如何继续训练

从最终模型继续训练：

```bash
RUN_NAME=g1_12dof_continue_$(date +%Y%m%d_%H%M%S) \
NUM_ENVS=512 \
ITERS=30000 \
LOGGER=tensorboard \
./g1_12dof_mjlab_train.sh train-bg \
  --agent.resume=True \
  --agent.load-run='.*g1_12dof_stride_extend_stairs_20260508_111519' \
  --agent.load-checkpoint='model_109499.pt' \
  --agent.save-interval=500 \
  --agent.algorithm.learning-rate=0.00025
```

查看状态：

```bash
./g1_12dof_mjlab_train.sh status
```

看实时曲线：

```bash
http://localhost:6006
```

## 后续调参建议

当前模型的主要剩余问题：

- 仍有一点小碎步倾向，`foot_x_separation_mean` 约 `0.113m`，目标是接近 `0.16m`。
- 后脚已经能到身体后方，但 `rear_foot_x_mean` 只有约 `-0.038m`，目标可以继续推到 `-0.06m` 或更后。
- 抬脚平均峰值约 `0.123m`，目标为 `0.16m`，继续训练可能还能提高，但需要防止变成夸张高抬腿。
- 台阶课程里 `low_open_stairs` 最终等级约 `1.47`，说明高台阶还比较难。

下一版可以考虑：

- 更直接奖励支撑腿后摆和髋关节后伸。
- 轻微提高 `sagittal_foot_separation` 权重，但不要过大，避免学成劈叉式不稳定 gait。
- 分阶段提高台阶高度，而不是从一开始就给 `0.14m`，让模型先稳定形成自然步幅。
- 对膝盖持续过度前屈加入轻微惩罚，减少“腿一直向前弯”的观感。
