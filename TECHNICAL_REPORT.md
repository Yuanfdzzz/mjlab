# G1 12DoF 感知行走训练技术报告

## 1. 项目目标

本项目基于 MJLab / MuJoCo Warp / RSL-RL，对 Unitree G1 12DoF 下肢模型训练前向感知行走策略。目标不是单纯让机器人向前移动，而是让它在离散、不连续地形上保持接近真实人形步态：

- 两脚交替迈步，不能交叉。
- 脚掌落地时尽量充分接触地面，避免脚尖点地或蹦跳。
- 膝盖保持轻微屈曲，避免僵直腿。
- 上下楼梯时保持直走，尽量不绕开、不侧偏。
- 逐步从平地、低台阶扩展到 12cm 台阶，再向 15cm 台阶过渡。

训练任务为：

```text
Mjlab-Velocity-EasyDiscontinuous-Unitree-G1-12Dof
```

当前较优保存模型：

```text
saved_models/g1_12dof_best_12cm_gait_248498_20260512/model_248498.pt
```

该模型是目前的 12cm 台阶基准版。后续 13.5cm / 15cm 训练均应以它作为可回滚基线。

## 2. 模型与训练框架

策略使用 RSL-RL 的 PPO 训练流程，环境由 MJLab 的 manager-based RL 环境组织。观测包括：

- 机器人本体状态：base 线速度、角速度、重力方向投影。
- 12 个关节的位置与速度。
- 上一步 action。
- 速度命令。
- 地形高度扫描 `height_scan`。
- critic 额外获得脚部高度、接触时间、接触状态与接触力。

网络结构保持原框架的 MLP actor-critic：

```text
actor obs: 393
critic obs: 405
hidden layers: 512 -> 256 -> 128
action dim: 12
```

关键训练脚本为：

```text
g1_12dof_mjlab_train.sh
```

常用后台训练形式：

```bash
NUM_ENVS=1536 ITERS=66000 RUN_NAME=<run_name> ./g1_12dof_mjlab_train.sh train-bg
```

## 3. 环境设计

### 3.1 训练地形

12DoF G1 使用专门的 gait terrain curriculum：

```python
_g1_12dof_gait_terrains_cfg()
```

当前训练阶段的地形包括：

| 地形 | 比例 | 高度范围 | 说明 |
| --- | ---: | ---: | --- |
| flat | 0.44 | 0 | 保留平地，防止步态被楼梯完全带偏 |
| low_up_stairs | 0.18 | 2-8cm | 低上楼梯 |
| low_down_stairs | 0.12 | 2-8cm | 低下楼梯 |
| mid_up_stairs | 0.18 | 8.5-13.5cm | 13.5cm 预备阶段上楼梯 |
| mid_down_stairs | 0.08 | 6-11.5cm | 中等下楼梯 |

在 12cm 最优版训练时，`mid_up_stairs` 上限为 12cm，抬脚目标为 16cm。当前继续训练阶段把上限提高到 13.5cm，抬脚目标提高到 18cm，为 15cm 台阶做准备。

### 3.2 展示地形

为了观察真实泛化效果，新增了只在 `play=True` 下使用的展示地形：

```python
_g1_12dof_demo_discontinuous_terrains_cfg()
```

展示地形采用 `curriculum=False`，每个地形块随机抽样，而不是一列一种地形。组成如下：

| 地形 | 比例 | 高度范围 |
| --- | ---: | ---: |
| short_flat_break | 0.08 | 平地 |
| low_up_stairs | 0.18 | 2-8cm |
| low_down_stairs | 0.16 | 2-8cm |
| mid_up_stairs | 0.24 | 8.5-13.5cm |
| mid_down_stairs | 0.18 | 6-11.5cm |
| uneven_up_stairs | 0.16 | 3.5-12cm 随机台阶 |

这样可视化时机器人会不断遇到随机上楼、下楼、短平地和不规则台阶，更适合作为压力测试，而不是只看某一列固定楼梯。

## 4. 奖励函数设计

整体奖励分为五类：任务跟踪、姿态稳定、足端接触、步态形态、失败约束。

### 4.1 任务跟踪与直线保持

主要目标是前向速度跟踪，但同时强约束偏航与横向漂移：

| 奖励项 | 作用 |
| --- | --- |
| `track_linear_velocity` | 跟踪前向速度命令 |
| `track_angular_velocity` | 抑制非期望 yaw 速度 |
| `forward_heading_alignment` | 鼓励身体朝前，不侧着走 |
| `lateral_drift` | 惩罚离初始直线路径的横向偏移 |
| `lateral_velocity` | 惩罚横向速度，防止上楼时斜着绕开 |

这部分是为了修复早期策略“看到楼梯就向侧面避开”的问题。

### 4.2 姿态稳定

| 奖励项 | 作用 |
| --- | --- |
| `upright` | 保持身体竖直 |
| `pose` | 保持接近合理默认姿态 |
| `body_ang_vel` | 惩罚躯干快速旋转 |
| `angular_momentum` | 抑制甩腿、翻跳 |
| `pelvis_height_floor` | 防止蹲得过低或塌陷 |

这里的重点是限制机器人用空翻、跳跃、扭腰等非步行方式刷前进奖励。

### 4.3 足端接触与抬脚

当前台阶阶段的抬脚目标：

```text
target_height = 0.18m
```

其中 12cm 最优模型使用过：

```text
target_height = 0.16m
```

几何关系按下面的经验式设计：

```text
foot_clearance_target >= stair_height_max + safety_margin
```

安全余量建议为 3-5cm。因此：

```text
12cm 台阶 -> 16cm 抬脚目标
13.5cm 台阶 -> 18cm 抬脚目标
15cm 台阶 -> 18-20cm 抬脚目标
```

相关奖励项：

| 奖励项 | 作用 |
| --- | --- |
| `foot_clearance` | 约束摆动脚与目标高度的偏差 |
| `foot_swing_height` | 约束落地前的峰值摆动高度 |
| `feet_height_in_air` | 鼓励腾空脚达到有效离地高度 |
| `left_foot_height_in_air` / `right_foot_height_in_air` | 单脚抬高奖励，用于修正左右不平衡 |
| `feet_swing_height_symmetry` | 惩罚左右脚抬高长期不对称 |

后期发现右脚长期低于左脚，因此在 13.5cm 阶段提高了右脚 air-time 与 height 权重。

### 4.4 交替步态与前后步幅

早期模型经常出现一条腿一直抬着、另一条腿蹬地，或者左右脚前后关系不切换。为此加入了一组 sagittal gait shaping：

| 奖励项 | 作用 |
| --- | --- |
| `alternating_foot_contacts` | 鼓励左右脚交替接触 |
| `alternating_foot_swings` | 鼓励左右脚交替摆动 |
| `sagittal_step_landing` | 落脚时要求摆动脚超过支撑脚 |
| `sagittal_foot_order_switch` | 鼓励前后脚顺序周期切换 |
| `sagittal_foot_order_stall` | 惩罚长期保持同一只脚在前 |
| `sagittal_foot_separation` | 惩罚前后脚距离太近 |
| `sagittal_stride_centering` | 防止整体步幅中心过度偏前或偏后 |
| `sagittal_front_rear_split` | 鼓励一脚在前、一脚在后 |
| `sagittal_foot_bounds` | 限制脚相对身体过度前伸或后甩 |

这组奖励是让模型从“会移动”走向“像走路”的关键。

### 4.5 防止交叉、跳跃和拖脚

| 奖励项 / 终止项 | 作用 |
| --- | --- |
| `foot_lateral_separation` | 保持左右脚横向间距，防止交叉 |
| `support_foot_flatness` | 支撑脚尽量平放 |
| `support_foot_ang_vel` | 支撑脚不要快速滚动 |
| `foot_slip` | 防止脚底打滑 |
| `soft_landing` | 减少落地冲击 |
| `feet_air_time_limit` | 防止单脚悬空太久 |
| `no_flight_phase` | 惩罚双脚同时离地，抑制跳跃 |
| `step_cadence_limit` | 限制过快步频 |
| `feet_air_time_symmetry` | 左右腾空时间对称 |
| `feet_contact_time_symmetry` | 左右支撑时间对称 |
| `foot_contact_stall` | 防止某只脚长期粘地 |

## 5. 终止条件与课程学习

### 5.1 终止条件

新增或调整的终止项包括：

| 终止项 | 作用 |
| --- | --- |
| `pelvis_too_low` | 骨盆过低时终止，防止倒地拖行 |
| `lateral_deviation` | 偏离直线过多时终止 |
| `right_foot_contact_stall` | 右脚长时间不离地时终止 |

需要注意的是，`pelvis_too_low` 在楼梯/下楼时不能设得过高，否则下楼过程中的世界坐标高度变化会被误判成倒地。

### 5.2 地形课程

新增课程函数：

```python
terrain_levels_forward_vel
```

它不是只看是否前进，而是同时看：

- 前向进展是否足够。
- 横向漂移是否在容忍范围内。

这样可以避免策略通过斜着走、绕开楼梯、走到地形边缘来升级难度。

## 6. 训练过程中犯过的错误

### 6.1 只奖励前进，导致异常步态

早期只要前向速度奖励足够强，模型会找到各种非人形步态：

- 一条腿一直抬着，另一条腿跳。
- 左脚动、右脚不动，机器人向右转圈。
- 右脚后蹬、左脚前伸，前后脚关系不切换。
- 双腿交叉，但仍能向前移动。

教训：对人形机器人来说，速度奖励不能单独定义“会走路”。必须加入足端相对位置、接触相位、左右对称、横向距离和支撑脚姿态约束。

### 6.2 台阶和抬脚目标不匹配

一度把楼梯高度提高得太快，而抬脚目标、安全余量和训练课程没有同步调整。结果包括：

- 脚刮台阶边缘。
- 策略用跳跃代替走路。
- 遇到上坡或台阶直接摔倒。
- 下楼时侧偏或失衡。

教训：台阶高度 `H_stair` 与抬脚目标 `H_clearance` 应满足：

```text
H_clearance >= H_stair + 0.03m ~ 0.05m
```

不能只提高地形难度而不提高足端 clearance。

### 6.3 地形难度跃迁过猛，导致模型训崩

有一轮直接把高台阶比例和高度推得太激进，早期日志出现：

- `pelvis_too_low` 大幅上升。
- episode length 很短。
- `both_feet_airborne_rate` 异常上升。

这说明模型开始用跳、塌、翻来探索，而不是保留原步态。该 run 被停止，并回滚到当时保存的较优模型。

教训：能力扩展应该分阶段进行。例如：

```text
12cm 稳定模型 -> 13.5cm 预备阶段 -> 15cm 阶段
```

每阶段都要低学习率微调，而不是突然重训。

### 6.4 可视化地形过于规则，误判泛化能力

早期展示地形常常是一列固定楼梯或固定难度地块。这样看起来会走，但不代表面对随机不连续地形也能走。

后来新增随机展示地形，使 `play=True` 时每块地形可能是上楼、下楼、随机楼梯或短平地，更能暴露刮脚、偏航、下楼失稳等问题。

### 6.5 未及时保存好模型，容易从坏 checkpoint 继续

训练中出现过“视觉上还不错，但后续训练训坏”的情况。如果没有明确保存当前最优版，就很容易从坏模型继续训练。

后来形成流程：

- 看到明显变好的模型，立即保存到 `saved_models/`。
- 保存模型、配置快照、README。
- 后续训练显式指定 `BASE_LOAD_RUN` 与 `BASE_LOAD_CHECKPOINT`。

## 7. 最终形成的创新点

### 7.1 从结果奖励转向形态奖励

普通速度跟踪只关心“到达速度”，本项目加入了大量形态约束，让策略必须通过接近自然步态的方式获得奖励：

- 前后脚交替。
- 左右脚不交叉。
- 支撑脚平放。
- 双脚不能同时长时间腾空。
- 步频不能过高。
- 膝盖保持屈曲。

这让 reward 从“移动奖励”变成了“感知地形上的人形行走奖励”。

### 7.2 结合足端相位和几何关系的 gait shaping

`sagittal_*` 系列奖励不是简单约束关节角，而是根据脚相对身体的位置、落脚事件、前后顺序切换来定义步态。这比强行指定固定关节轨迹更灵活，允许模型自己找到可行步态，同时避免一脚拖地或一脚独占。

### 7.3 面向楼梯的 clearance-geometry 设计

将楼梯高度与抬脚目标绑定：

```text
clearance target = stair max height + safety margin
```

这使地形课程和奖励目标同步增长，减少“台阶变高但脚不抬”的失配。

### 7.4 横向漂移参与课程升级

课程学习不再只看前进距离，还检查横向偏移。这样可以惩罚策略通过绕开台阶来升级，迫使它真正适应不连续地形。

### 7.5 训练地形与展示地形分离

训练使用 curriculum，保证学习稳定；展示使用随机不连续地形，保证测试足够苛刻。这种分离让训练和评估各自服务于不同目标：

- 训练：循序渐进，不把策略打崩。
- 展示：随机混合，暴露真实弱点。

## 8. 当前模型状态

### 8.1 12cm 最优基线

保存路径：

```text
saved_models/g1_12dof_best_12cm_gait_248498_20260512/model_248498.pt
```

保存时观察到的关键指标：

| 指标 | 数值 |
| --- | ---: |
| Mean reward | 101.36 |
| Mean episode length | 481.95 |
| peak foot height mean | 约 13.4cm |
| swing foot height mean | 约 10.6cm |
| left/right foot height mean | 约 6.4cm / 5.0cm |
| foot crossing rate | 约 0 |
| both-feet-airborne rate | 约 1.7% |

判断：这是目前最稳的 12cm 台阶基线，但还不能直接稳定上 15cm，因为平均峰值高度距离 15cm 台阶所需的 18-20cm clearance 仍不足。

### 8.2 当前进行中的 13.5cm 预备训练

当前 run：

```text
g1_12dof_stage6_135cm_to_15cm_prep_20260512_121623
```

配置：

| 项 | 值 |
| --- | --- |
| 起点 | `model_248498.pt` |
| num envs | 1536 |
| learning rate | 1.5e-5 |
| mid up stairs | 8.5-13.5cm |
| clearance target | 18cm |

截至最近日志，训练仍在进行。早期/中期指标显示：

- `foot_crossing_rate` 仍接近 0。
- `both_feet_airborne_rate` 约 2% 左右，暂未变成明显跳跃。
- `peak_height_mean` 已提升到约 14.3cm。
- 右脚高度有所改善，但仍低于左脚。
- `lateral_deviation` 仍偏高，是下一步需要继续压的风险项。

## 9. 后续建议

1. 当前 13.5cm 阶段不要急于改到 15cm，应先确认可视化中没有跳跃化、右脚刮台阶、上楼侧偏。

2. 如果 13.5cm 阶段稳定，可以再开 15cm 阶段：

```text
mid_up_stairs: 0.110-0.150m
foot target: 0.19-0.20m
learning rate: 1e-5 到 1.5e-5
```

3. 若出现跳跃，应优先增强：

- `no_flight_phase`
- `feet_air_time_limit`
- `step_cadence_limit`
- `angular_momentum`

4. 若出现右脚刮台阶，应继续增强：

- `right_foot_height_in_air`
- `right_foot_air_time`
- `feet_swing_height_symmetry`

5. 若出现上楼绕开或下楼偏右，应增强：

- `lateral_drift`
- `lateral_velocity`
- `forward_heading_alignment`
- `terrain_levels_forward_vel` 中的 lateral tolerance

## 10. 总结

这个项目的核心经验是：感知行走不是把速度奖励放大就能解决的问题。人形机器人会非常善于利用奖励漏洞，用跳、蹬、拖、交叉、侧移等方式完成表面任务。因此最终有效的方案，是把地形课程、足端几何、接触相位、左右对称、直线约束和失败终止统一起来。

目前 12cm 台阶模型已经形成一个较可靠的基线；正在进行的 13.5cm 预备阶段是在这个基线上扩展 clearance 和楼梯能力。后续冲 15cm 时，最重要的是保留当前自然步态，而不是让模型重新学出“能上楼但不像走路”的投机策略。
