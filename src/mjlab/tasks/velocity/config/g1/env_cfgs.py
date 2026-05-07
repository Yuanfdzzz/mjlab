"""Unitree G1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_12DOF_ACTION_SCALE,
  G1_ACTION_SCALE,
  get_g1_12dof_robot_cfg,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import (
  TerrainGeneratorCfg,
  box_random_grid,
  flat,
  hf_pyramid_slope,
  narrow_beams,
  nested_rings,
  open_stairs,
  random_stairs,
  stepping_stones,
)


def _g1_discontinuous_terrains_cfg() -> TerrainGeneratorCfg:
  """Terrain set for humanoid perceptive locomotion on discontinuous footholds."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=8,
    curriculum=True,
    sub_terrains={
      # Keep some easy cells so early curriculum has recoverable samples.
      "flat": flat(proportion=0.10),
      "easy_open_stairs": open_stairs(
        proportion=0.15,
        step_height_range=(0.04, 0.10),
        step_width_range=(0.45, 0.80),
      ),
      "random_stairs": random_stairs(
        proportion=0.15,
        step_width=0.55,
        step_height_range=(0.05, 0.18),
      ),
      "stepping_stones": stepping_stones(
        proportion=0.25,
        stone_size_range=(0.35, 0.70),
        stone_distance_range=(0.12, 0.40),
        stone_height=0.12,
        stone_height_variation=0.12,
        stone_size_variation=0.18,
        displacement_range=0.12,
      ),
      "narrow_beams": narrow_beams(
        proportion=0.10,
        num_beams=10,
        beam_width_range=(0.18, 0.45),
        beam_height=0.16,
        spacing=0.65,
      ),
      "nested_rings": nested_rings(
        proportion=0.10,
        num_rings=6,
        ring_width_range=(0.25, 0.55),
        gap_range=(0.08, 0.28),
        height_range=(0.08, 0.28),
      ),
      "box_random_grid": box_random_grid(
        proportion=0.15,
        grid_width=0.35,
        grid_height_range=(0.02, 0.22),
      ),
    },
    add_lights=True,
  )


def _g1_easy_discontinuous_terrains_cfg() -> TerrainGeneratorCfg:
  """Intro curriculum with stairs and small gaps before harder footholds."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=5,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.25),
      "low_open_stairs": open_stairs(
        proportion=0.30,
        step_height_range=(0.02, 0.08),
        step_width_range=(0.65, 0.95),
      ),
      "low_random_stairs": random_stairs(
        proportion=0.20,
        step_width=0.75,
        step_height_range=(0.02, 0.10),
      ),
      "small_gap_platforms": stepping_stones(
        proportion=0.20,
        stone_size_range=(0.65, 0.95),
        stone_distance_range=(0.02, 0.14),
        stone_height=0.06,
        stone_height_variation=0.04,
        stone_size_variation=0.06,
        displacement_range=0.03,
      ),
      "low_box_grid": box_random_grid(
        proportion=0.05,
        grid_width=0.55,
        grid_height_range=(0.01, 0.08),
      ),
    },
    add_lights=True,
  )


def _g1_12dof_gait_terrains_cfg() -> TerrainGeneratorCfg:
  """Early gait curriculum for the leg-only G1 model.

  The 12DoF lower-body policy needs to learn forward steps before it can survive
  foothold gaps. Keep the first retraining stage dominated by flat ground and
  gentle slopes, with low stairs only as a mild foot-clearance signal.
  """
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=3,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.50),
      "gentle_up_slope": hf_pyramid_slope(
        proportion=0.35,
        slope_range=(0.0, 0.25),
        platform_width=2.0,
      ),
      "low_open_stairs": open_stairs(
        proportion=0.15,
        step_height_range=(0.015, 0.055),
        step_width_range=(0.75, 1.05),
      ),
    },
    add_lights=True,
  )


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_discontinuous_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 discontinuous-terrain perceptive locomotion config."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = _g1_discontinuous_terrains_cfg()
  cfg.scene.terrain.max_init_terrain_level = 2

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern.size = (2.2, 1.4)
      sensor.pattern.resolution = 0.10
      sensor.max_distance = 6.0
      cfg.observations["actor"].terms["height_scan"].scale = 1 / sensor.max_distance
      cfg.observations["critic"].terms["height_scan"].scale = 1 / sensor.max_distance

  # Discontinuous footholds need slower early commands than rough slopes/stairs.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.4, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.4, 1.0), "ang_vel_z": (-0.4, 0.4)},
      {"step": 5000 * 24, "lin_vel_x": (-0.8, 1.5), "ang_vel_z": (-0.5, 0.5)},
      {"step": 10000 * 24, "lin_vel_x": (-1.2, 2.0), "ang_vel_z": (-0.7, 0.7)},
    ]

  cfg.rewards["foot_clearance"].params["target_height"] = 0.16
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.16
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_swing_height"].weight = -0.2
  cfg.rewards["foot_slip"].weight = -0.2
  cfg.rewards["soft_landing"].weight = -2e-5

  if play:
    cfg.curriculum = {}
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 4
      cfg.scene.terrain.terrain_generator.num_rows = 4
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_easy_discontinuous_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 easy discontinuous-terrain walking config."""
  cfg = unitree_g1_discontinuous_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _g1_easy_discontinuous_terrains_cfg()
  cfg.scene.terrain.max_init_terrain_level = 1

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.2, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.25, 0.25)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.2, 0.8), "ang_vel_z": (-0.25, 0.25)},
      {"step": 5000 * 24, "lin_vel_x": (-0.4, 1.0), "ang_vel_z": (-0.4, 0.4)},
      {"step": 10000 * 24, "lin_vel_x": (-0.8, 1.5), "ang_vel_z": (-0.5, 0.5)},
    ]

  cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.10

  if play and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.num_cols = 3
    cfg.scene.terrain.terrain_generator.num_rows = 3

  return cfg


def unitree_g1_12dof_discontinuous_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 12DoF discontinuous-terrain perceptive locomotion config."""
  cfg = unitree_g1_discontinuous_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_12dof_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_12DOF_ACTION_SCALE

  cfg.viewer.body_name = "pelvis"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = None
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

  # The 12DoF XML only exposes the leg joints; keep posture priors focused there.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.30,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.10,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.50,
    r".*hip_roll.*": 0.20,
    r".*hip_yaw.*": 0.20,
    r".*knee.*": 0.60,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }

  # The leg-only model can otherwise learn a low, sliding gait: velocity reward is
  # reachable without clear swing phases, while clearance terms only shape feet
  # once they are already moving. Add a modest swing incentive for moving commands.
  cfg.rewards["air_time"].weight = 0.45
  cfg.rewards["air_time"].params["command_threshold"] = 0.10
  cfg.rewards["air_time"].params["threshold_min"] = 0.04
  cfg.rewards["air_time"].params["threshold_max"] = 0.35
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_swing_height"].weight = -0.4
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.terminations["nan_detection"] = TerminationTermCfg(
    func=envs_mdp.nan_detection,
    time_out=False,
  )
  for obs_group in cfg.observations.values():
    obs_group.nan_policy = "sanitize"

  return cfg


def unitree_g1_12dof_easy_discontinuous_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 12DoF easy discontinuous-terrain walking config."""
  cfg = unitree_g1_easy_discontinuous_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_12dof_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _g1_12dof_gait_terrains_cfg()
  cfg.scene.terrain.max_init_terrain_level = 0

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_12DOF_ACTION_SCALE

  cfg.viewer.body_name = "pelvis"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = None
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.70,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.20,
    r".*knee.*": 0.85,
    r".*ankle_pitch.*": 0.45,
    r".*ankle_roll.*": 0.16,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.90,
    r".*hip_roll.*": 0.30,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 1.00,
    r".*ankle_pitch.*": 0.55,
    r".*ankle_roll.*": 0.20,
  }

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.45, 0.90)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.init_velocity_prob = 0.0
  twist_cmd.resampling_time_range = (4.0, 7.0)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot")
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.45, 0.75), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
      {"step": 10_000_000, "lin_vel_x": (0.45, 0.85), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    ]

  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.22
  cfg.rewards["track_angular_velocity"].weight = 0.4
  cfg.rewards["upright"].weight = 0.8
  cfg.rewards["pose"].weight = 0.25
  cfg.rewards["air_time"].weight = 0.6
  cfg.rewards["air_time"].params["command_threshold"] = 0.10
  cfg.rewards["air_time"].params["threshold_min"] = 0.08
  cfg.rewards["air_time"].params["threshold_max"] = 0.38
  cfg.rewards["foot_clearance"].weight = -0.20
  cfg.rewards["foot_clearance"].params["target_height"] = 0.12
  cfg.rewards["foot_swing_height"].weight = -0.8
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
  cfg.rewards["foot_slip"].weight = -0.35
  cfg.rewards["action_rate_l2"].weight = -0.015
  cfg.rewards["alternating_foot_contacts"] = RewardTermCfg(
    func=mdp.alternating_foot_contacts,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.10,
    },
  )
  cfg.rewards["feet_air_time_limit"] = RewardTermCfg(
    func=mdp.feet_air_time_limit,
    weight=-3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "max_air_time": 0.42,
      "command_name": "twist",
      "command_threshold": 0.10,
    },
  )
  cfg.rewards["feet_air_time_symmetry"] = RewardTermCfg(
    func=mdp.feet_air_time_symmetry,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.10,
    },
  )
  cfg.rewards["no_flight_phase"] = RewardTermCfg(
    func=mdp.no_flight_phase,
    weight=-0.8,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.10,
    },
  )
  cfg.terminations["nan_detection"] = TerminationTermCfg(
    func=envs_mdp.nan_detection,
    time_out=False,
  )
  cfg.terminations["pelvis_too_low"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.55,
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
    },
    time_out=False,
  )
  for obs_group in cfg.observations.values():
    obs_group.nan_policy = "sanitize"

  if play:
    twist_cmd.ranges.lin_vel_x = (0.45, 0.45)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
