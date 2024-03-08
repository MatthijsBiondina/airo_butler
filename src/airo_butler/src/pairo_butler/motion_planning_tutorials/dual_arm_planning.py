import time
import numpy as np
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pairo_butler.utils.tools import pyout
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.drake.visualization import publish_dual_arm_joint_path

# --------------------------------------------------------------------------------

robot_diagram_builder = RobotDiagramBuilder()
meshcat = add_meshcat_to_builder(robot_diagram_builder)
arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(robot_diagram_builder)
diagram, context = finish_build(robot_diagram_builder, meshcat)

# --------------------------------------------------------------------------------

collision_checker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[*arm_indices, *gripper_indices],
    edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
    env_collision_padding=0.005,
    self_collision_padding=0.005,
)

# --------------------------------------------------------------------------------

start_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0])
start_joints_right = np.deg2rad([-136, -116, -110, -133, 40, 0])
# start_joints_left = np.deg2rad([90, -135, 95, -50, -90, -90])
# start_joints_right = np.deg2rad([-90, -45, -95, -130, 90, 90])

start_joints = np.concatenate([start_joints_left, start_joints_right])
collision_checker.CheckConfigCollisionFree(start_joints)

# --------------------------------------------------------------------------------
time.sleep(5)
plant = diagram.plant()
plant_context = plant.GetMyContextFromRoot(context)

arm_left_index, arm_right_index = arm_indices
plant.SetPositions(plant_context, arm_left_index, start_joints_left)
plant.SetPositions(plant_context, arm_right_index, start_joints_right)
diagram.ForcedPublish(context)
time.sleep(3)
# --------------------------------------------------------------------------------

home_joints_left = np.deg2rad([180, -120, 60, -30, -90, -90])
home_joints_right = np.deg2rad([-180, -60, -60, -150, 90, 90])

home_joints = np.concatenate([home_joints_left, home_joints_right])
collision_checker.CheckConfigCollisionFree(home_joints)

# --------------------------------------------------------------------------------

plant.SetPositions(plant_context, arm_left_index, home_joints_left)
plant.SetPositions(plant_context, arm_right_index, home_joints_right)
diagram.ForcedPublish(context)

# --------------------------------------------------------------------------------

planner = DualArmOmplPlanner(
    collision_checker.CheckConfigCollisionFree, max_planning_time=10
)

# --------------------------------------------------------------------------------

path = planner.plan_to_joint_configuration(
    start_joints_left, start_joints_right, home_joints_left, home_joints_right
)

# --------------------------------------------------------------------------------

from cloth_tools.drake.visualization import publish_dual_arm_trajectory
from cloth_tools.path.execution import time_parametrize_toppra


joint_trajectory, time_trajectory = time_parametrize_toppra(path, plant)


publish_dual_arm_trajectory(
    joint_trajectory,
    time_trajectory,
    meshcat,
    diagram,
    context,
    arm_left_index,
    arm_right_index,
)
# publish_dual_arm_joint_path(path, 5.0, meshcat, diagram, context, arm_left_index, arm_right_index)

# --------------------------------------------------------------------------------

path2 = planner.plan_to_joint_configuration(
    home_joints_left, home_joints_right, start_joints_left, start_joints_right
)

# --------------------------------------------------------------------------------

joint_trajectory2, time_trajectory2 = time_parametrize_toppra(path2, plant)
publish_dual_arm_trajectory(
    joint_trajectory2,
    time_trajectory2,
    meshcat,
    diagram,
    context,
    arm_left_index,
    arm_right_index,
)

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
pyout("Done!")
time.sleep(300)
