import time
import numpy as np
from typing import List
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_ur5e_and_table_to_builder
from cloth_tools.drake.visualization import add_meshcat_triad
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from ur_analytic_ik import ur5e

# --------------------------------------------------------------------------------

tcp_transform = np.identity(4)
tcp_transform[2, 3] = 0.175

# --------------------------------------------------------------------------------

robot_diagram_builder = RobotDiagramBuilder()
meshcat = add_meshcat_to_builder(robot_diagram_builder)
arm_index, gripper_index = add_ur5e_and_table_to_builder(robot_diagram_builder)
diagram, context = finish_build(robot_diagram_builder, meshcat)
plant = diagram.plant()
plant_context = plant.GetMyContextFromRoot(context)

collision_checker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[arm_index, gripper_index],
    edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
    env_collision_padding=0.005,
    self_collision_padding=0.005,
)

# --------------------------------------------------------------------------------

start_joints = np.deg2rad([0, -90, -90, -90, 90, 0])

plant = diagram.plant()
plant_context = plant.GetMyContextFromRoot(context)

plant.SetPositions(plant_context, arm_index, start_joints)
diagram.ForcedPublish(context)

time.sleep(5)

# --------------------------------------------------------------------------------

transform = RigidTransform(p=[-0.15, 0.0, 0.2], rpy=RollPitchYaw([np.pi, 0, 0]))
tcp_pose_0 = np.ascontiguousarray(transform.GetAsMatrix4())

add_meshcat_triad(meshcat, "TCP Frame", X_W_Triad=transform)

# --------------------------------------------------------------------------------


def inverse_kinematics_fn(
    tcp_pose: HomogeneousMatrixType,
) -> List[JointConfigurationType]:
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(tcp_pose, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


# --------------------------------------------------------------------------------

from cloth_tools.ompl.single_arm_planner import SingleArmOmplPlanner


planner = SingleArmOmplPlanner(
    collision_checker.CheckConfigCollisionFree, inverse_kinematics_fn
)
path = planner.plan_to_tcp_pose(start_joints, tcp_pose_0)

# --------------------------------------------------------------------------------

from cloth_tools.drake.visualization import publish_joint_path

publish_joint_path(path, 5.0, meshcat, diagram, context, arm_index)

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

time.sleep(60)
