"""Unitree G1 12DoF constants.

This asset adapts the G1 12DoF MJCF from the sibling ``rsl_rl/unitree_rl_gym``
checkout so it can be used as an mjlab entity.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_DEFAULT_G1_12DOF_XML = Path(__file__).parent / "xmls" / "g1_12dof.xml"

G1_12DOF_XML = Path(os.environ.get("MJLAB_G1_12DOF_XML", _DEFAULT_G1_12DOF_XML))
assert G1_12DOF_XML.exists(), (
  f"G1 12DoF XML not found: {G1_12DOF_XML}. "
  "Set MJLAB_G1_12DOF_XML to override the path."
)


def get_spec() -> mujoco.MjSpec:
  """Load the source MJCF and adapt it for mjlab position actions."""
  spec = mujoco.MjSpec.from_file(str(G1_12DOF_XML))

  # The source XML contains effort motors for the standalone deploy script.
  # mjlab will add its own PD motors from the articulation config below.
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  pelvis = spec.body("pelvis")
  pelvis.add_site(name="imu_in_pelvis", pos=[0.04525, 0.0, -0.08339], size=[0.01])
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="pelvis",
  )

  # Add sites used by mjlab's foot-height sensor and foot-placement rewards.
  left_foot_body = spec.body("left_ankle_roll_link")
  right_foot_body = spec.body("right_ankle_roll_link")
  left_foot_body.add_site(name="left_foot", pos=[0.04, 0.0, -0.03], size=[0.01])
  right_foot_body.add_site(name="right_foot", pos=[0.04, 0.0, -0.03], size=[0.01])

  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.80),
  joint_pos={
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

G1_12DOF_HIP_YAW_PITCH_ACTUATOR = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", ".*_hip_pitch_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=88.0,
  armature=0.01,
  frictionloss=0.1,
)
G1_12DOF_HIP_ROLL_ACTUATOR = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=139.0,
  armature=0.01,
  frictionloss=0.1,
)
G1_12DOF_KNEE_ACTUATOR = IdealPdActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=139.0,
  armature=0.01,
  frictionloss=0.1,
)
G1_12DOF_ANKLE_ACTUATOR = IdealPdActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=40.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
  frictionloss=0.1,
)

G1_12DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_12DOF_HIP_YAW_PITCH_ACTUATOR,
    G1_12DOF_HIP_ROLL_ACTUATOR,
    G1_12DOF_KNEE_ACTUATOR,
    G1_12DOF_ANKLE_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

G1_12DOF_ACTION_SCALE = 0.25


def get_g1_12dof_robot_cfg() -> EntityCfg:
  """Get a fresh G1 12DoF robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=G1_12DOF_ARTICULATION,
    sort_actuators=True,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_12dof_robot_cfg())
  viewer.launch(robot.spec.compile())
