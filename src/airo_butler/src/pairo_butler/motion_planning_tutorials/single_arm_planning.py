import time
import numpy as np
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pairo_butler.utils.tools import pyout
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_ur5e_and_table_to_builder
from cloth_tools.drake.visualization import publish_joint_path
from cloth_tools.ompl.single_arm_planner import SingleArmOmplPlanner

# ---------------------------------------------------------------------------

robot_diagram_builder = (
    RobotDiagramBuilder()
)  # time_step=0.001 even when I set timestep I get the mimic joint warning
meshcat = add_meshcat_to_builder(robot_diagram_builder)
arm_index, gripper_index = add_ur5e_and_table_to_builder(robot_diagram_builder)

pyout(arm_index, gripper_index)

# ---------------------------------------------------------------------------

diagram, context = finish_build(robot_diagram_builder, meshcat)
plant = diagram.plant()
plant_context = plant.GetMyContextFromRoot(context)

# ---------------------------------------------------------------------------

collision_checker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[arm_index, gripper_index],
    edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
    env_collision_padding=0.005,
    self_collision_padding=0.005,
)

# ---------------------------------------------------------------------------

start_joints = np.deg2rad([0, -90, -90, -90, 90, 0])
pyout(collision_checker.CheckConfigCollisionFree(start_joints))

# ---------------------------------------------------------------------------


# To visualize the start pose in meshcat
plant.SetPositions(plant_context, arm_index, start_joints)
diagram.ForcedPublish(context)

# ---------------------------------------------------------------------------

goal_joints = np.deg2rad([180, -135, 95, -50, -90, -90])
pyout(collision_checker.CheckConfigCollisionFree(goal_joints))

# ---------------------------------------------------------------------------

planner = SingleArmOmplPlanner(
    is_state_valid_fn=collision_checker.CheckConfigCollisionFree
)

# ---------------------------------------------------------------------------


path = planner.plan_to_joint_configuration(start_joints, goal_joints)

# ---------------------------------------------------------------------------

from cloth_tools.path.execution import calculate_path_array_duration

time.sleep(5)

duration = calculate_path_array_duration(np.array(path))
print(duration)

publish_joint_path(path, duration, meshcat, diagram, context, arm_index)

# ---------------------------------------------------------------------------

time.sleep(3)

from typing import List, Tuple

from airo_typing import JointConfigurationType
from pydrake.trajectories import PiecewisePolynomial, Trajectory
from pydrake.multibody.optimization import CalcGridPointsOptions, Toppra


def time_parametrize(
    path: List[JointConfigurationType], duration
) -> Tuple[Trajectory, Trajectory]:
    # original path q(s) with s = s(t) hence q(s(t))
    plant = diagram.plant()
    # start_time = 0
    # end_time = duration

    # q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    #     np.linspace(start_time, end_time, len(path)), path_np.T
    # )

    path_array = np.array(path)

    if len(path_array) >= 3:
        q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.linspace(0.0, 1.0, len(path_array)), path_array.T
        )
    else:
        q_traj = PiecewisePolynomial.FirstOrderHold([0.0, 1.0], path_array.T)

    gridpoints = Toppra.CalcGridPoints(q_traj, CalcGridPointsOptions())
    toppra = Toppra(q_traj, plant, gridpoints)
    toppra.AddJointAccelerationLimit([-1.2] * 6, [1.2] * 6)
    toppra.AddJointVelocityLimit([-1] * 6, [1] * 6)
    t_traj = toppra.SolvePathParameterization()

    return q_traj, t_traj


q_traj, t_traj = time_parametrize(path, duration)


from typing import List, Tuple

import numpy as np
from airo_typing import JointConfigurationType
from loguru import logger
from pydrake.geometry import Cylinder, Meshcat, Rgba
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagram
from pydrake.systems.framework import Context


def publish_q_traj_t_traj(
    q_traj: Trajectory,
    t_traj: Trajectory,
    meshcat: Meshcat,
    diagram: RobotDiagram,
    context: Context,
    arm_index: ModelInstanceIndex,
) -> None:
    plant = diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)

    meshcat.StartRecording(set_visualizations_while_recording=False)

    duration = t_traj.end_time()
    fps = 60.0
    frames = duration * fps

    for t in np.linspace(0, duration, int(np.ceil(frames))):
        context.SetTime(t)
        q = q_traj.value(t_traj.value(t).item())
        plant.SetPositions(plant_context, arm_index, q)
        diagram.ForcedPublish(context)

    meshcat.StopRecording()
    meshcat.PublishRecording()


publish_q_traj_t_traj(q_traj, t_traj, meshcat, diagram, context, arm_index)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

pyout()
time.sleep(300)
