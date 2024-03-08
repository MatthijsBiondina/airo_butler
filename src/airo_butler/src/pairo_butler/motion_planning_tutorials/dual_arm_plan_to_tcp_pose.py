import time
import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.drake.visualization import (
    publish_dual_arm_joint_path,
    publish_ik_solutions,
    add_meshcat_triad,
)

# --------------------------------------------------------------------------------

tcp_transform = np.identity(4)
tcp_transform[2, 3] = 0.175

# For table mounted setup: shoulder outside this range will almost certainly collide with the table
# For all UR robots: elbow constrainted -160 to 160 due to self-collision
joint_bounds_lower = np.deg2rad([-360, -180, -160, -360, -360, -360])
joint_bounds_upper = np.deg2rad([360, 0, 160, 360, 360, 360])
joint_bounds = (joint_bounds_lower, joint_bounds_upper)

# --------------------------------------------------------------------------------

# Creating the default scene
robot_diagram_builder = RobotDiagramBuilder()
meshcat = add_meshcat_to_builder(robot_diagram_builder)
arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(robot_diagram_builder)
diagram, context = finish_build(robot_diagram_builder, meshcat)

collision_checker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[*arm_indices, *gripper_indices],
    edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
    env_collision_padding=0.005,
    self_collision_padding=0.005,
)

is_state_valid_fn = collision_checker.CheckConfigCollisionFree

# --------------------------------------------------------------------------------

home_joints_left = np.deg2rad([180, -120, 60, -30, -90, -90])
home_joints_right = np.deg2rad([-180, -60, -60, -150, 90, 90])

# --------------------------------------------------------------------------------

plant = diagram.plant()
plant_context = plant.GetMyContextFromRoot(context)

arm_left_index, arm_right_index = arm_indices
plant.SetPositions(plant_context, arm_left_index, home_joints_left)
plant.SetPositions(plant_context, arm_right_index, home_joints_right)
diagram.ForcedPublish(context)

# --------------------------------------------------------------------------------

transform_0 = RigidTransform(p=[0, 0, 0.35], rpy=RollPitchYaw([np.pi, 0, 0]))
tcp_pose_0 = np.ascontiguousarray(transform_0.GetAsMatrix4())

add_meshcat_triad(meshcat, "TCP Frame left", X_W_Triad=transform_0)

transform_1 = RigidTransform(
    p=[0.15, 0, 0.3], rpy=RollPitchYaw([np.pi / 2, 0, np.pi / 2])
)
tcp_pose_1 = np.ascontiguousarray(transform_1.GetAsMatrix4())

add_meshcat_triad(meshcat, "TCP Frame right", X_W_Triad=transform_1)

time.sleep(5)

# --------------------------------------------------------------------------------

from functools import partial
from typing import List
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from ur_analytic_ik import ur5e

from cloth_tools.drake.scenes import X_W_L_DEFAULT, X_W_R_DEFAULT, X_CB_B

X_W_LCB = X_W_L_DEFAULT @ X_CB_B.inverse()
X_W_RCB = X_W_R_DEFAULT @ X_CB_B.inverse()


def inverse_kinematics_in_world_fn(
    tcp_pose: HomogeneousMatrixType, X_W_CB: HomogeneousMatrixType
) -> List[JointConfigurationType]:
    X_W_TCP = tcp_pose
    X_CB_W = np.linalg.inv(X_W_CB)
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_CB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


inverse_kinematics_left_fn = partial(
    inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB.GetAsMatrix4()
)
inverse_kinematics_right_fn = partial(
    inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB.GetAsMatrix4()
)

# --------------------------------------------------------------------------------

solutions_left = inverse_kinematics_left_fn(tcp_pose_0)
# publish_ik_solutions(solutions_left, 2.0, meshcat, diagram, context, arm_left_index)

# --------------------------------------------------------------------------------

solutions_right = inverse_kinematics_right_fn(tcp_pose_1)
# publish_ik_solutions(solutions_right, 2.0, meshcat, diagram, context, arm_right_index)

# --------------------------------------------------------------------------------

planner = DualArmOmplPlanner(
    is_state_valid_fn,
    inverse_kinematics_left_fn,
    inverse_kinematics_right_fn,
    joint_bounds_left=joint_bounds,
    joint_bounds_right=joint_bounds,
)

time.sleep(10)

# --------------------------------------------------------------------------------

path = planner.plan_to_tcp_pose(home_joints_left, home_joints_right, tcp_pose_0, None)
publish_dual_arm_joint_path(path, 2.0, meshcat, diagram, context, *arm_indices)

# --------------------------------------------------------------------------------

time.sleep(5)

path = planner.plan_to_tcp_pose(home_joints_left, home_joints_right, None, tcp_pose_1)
publish_dual_arm_joint_path(path, 2.0, meshcat, diagram, context, *arm_indices)

# --------------------------------------------------------------------------------

time.sleep(5)


path = planner.plan_to_tcp_pose(
    home_joints_left, home_joints_right, tcp_pose_0, tcp_pose_1
)
publish_dual_arm_joint_path(path, 2.0, meshcat, diagram, context, *arm_indices)

time.sleep(5)


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


time.sleep(60)
