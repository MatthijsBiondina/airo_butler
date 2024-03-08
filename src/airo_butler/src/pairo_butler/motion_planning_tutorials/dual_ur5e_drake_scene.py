import time
import numpy as np
import airo_models
from pydrake.geometry import Meshcat
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.geometry import MeshcatVisualizer
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pairo_butler.utils.tools import pyout
from cloth_tools.drake.visualization import add_meshcat_triad
from cloth_tools.urdf.robotiq import create_static_robotiq_2f_85_urdf

robot_diagram_builder = (
    RobotDiagramBuilder()
)  # time_step=0.001 even when I set timestep I get the mimic joint warning
scene_graph = robot_diagram_builder.scene_graph()
plant = robot_diagram_builder.plant()
builder = robot_diagram_builder.builder()
parser = robot_diagram_builder.parser()
parser.SetAutoRenaming(True)

# Add visualizer
meshcat = Meshcat()
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

## Example of how to build a scene, this is does the same as the add_dual_ur5e_and_table_to_builder() function
# Load URDF files
ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
robotiq_urdf_path = create_static_robotiq_2f_85_urdf()

table_thickness = 0.2
table_urdf_path = airo_models.box_urdf_path((2.0, 2.4, table_thickness), "table")
wall_thickness = 0.2
wall_back_urdf_path = airo_models.box_urdf_path((wall_thickness, 2.7, 2.0), "wall_back")
wall_left_urdf_path = airo_models.box_urdf_path((2.0, wall_thickness, 2.0), "wall_left")
wall_right_urdf_path = airo_models.box_urdf_path(
    (2.0, wall_thickness, 2.0), "wall_right"
)

arm_left_index = parser.AddModels(ur5e_urdf_path)[0]
arm_right_index = parser.AddModels(ur5e_urdf_path)[0]
gripper_left_index = parser.AddModels(robotiq_urdf_path)[0]
gripper_right_index = parser.AddModels(robotiq_urdf_path)[0]

table_index = parser.AddModels(table_urdf_path)[0]
wall_back_index = parser.AddModels(wall_back_urdf_path)[0]
wall_left_index = parser.AddModels(wall_left_urdf_path)[0]
wall_right_index = parser.AddModels(wall_right_urdf_path)[0]

# Weld some frames together
world_frame = plant.world_frame()
arm_left_frame = plant.GetFrameByName("base_link", arm_left_index)
arm_right_frame = plant.GetFrameByName("base_link", arm_right_index)
arm_left_tool_frame = plant.GetFrameByName("tool0", arm_left_index)
arm_right_tool_frame = plant.GetFrameByName("tool0", arm_right_index)
gripper_left_frame = plant.GetFrameByName("base_link", gripper_left_index)
gripper_right_frame = plant.GetFrameByName("base_link", gripper_right_index)

table_frame = plant.GetFrameByName("base_link", table_index)
wall_back_frame = plant.GetFrameByName("base_link", wall_back_index)
wall_left_frame = plant.GetFrameByName("base_link", wall_left_index)
wall_right_frame = plant.GetFrameByName("base_link", wall_right_index)

arm_y = 0.45
arm_left_transform = RigidTransform(
    rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, arm_y, 0]
)
arm_right_transform = RigidTransform(
    rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -arm_y, 0]
)
robotiq_ur_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, 0, 0])
table_transform = RigidTransform(p=[0, 0, -table_thickness / 2])
wall_back_transform = RigidTransform(p=[0.9 + wall_thickness / 2, 0, 0])
wall_left_transform = RigidTransform(p=[0, arm_y + 0.7 + wall_thickness / 2, 0])
wall_right_transform = RigidTransform(p=[0, -arm_y - 0.7 - wall_thickness / 2, 0])

plant.WeldFrames(world_frame, arm_left_frame, arm_left_transform)
plant.WeldFrames(world_frame, arm_right_frame, arm_right_transform)
plant.WeldFrames(arm_left_tool_frame, gripper_left_frame, robotiq_ur_transform)
plant.WeldFrames(arm_right_tool_frame, gripper_right_frame, robotiq_ur_transform)
plant.WeldFrames(world_frame, table_frame, table_transform)
plant.WeldFrames(world_frame, wall_back_frame, wall_back_transform)
plant.WeldFrames(world_frame, wall_left_frame, wall_left_transform)
plant.WeldFrames(world_frame, wall_right_frame, wall_right_transform)

add_meshcat_triad(meshcat, "World", length=0.3)

# --------------------------------------------------

# A diagram is needed in the constructor of the SceneGraphCollisionChecker
# However, calling .Build() prevents us from adding more models, e.g. runtime obstacles
diagram = robot_diagram_builder.Build()

# Create default contexts ~= state
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
diagram.ForcedPublish(context)

# --------------------------------------------------

nr_pos_left, nr_pos_right = plant.num_positions(
    gripper_left_index
), plant.num_positions(gripper_right_index)
pyout(nr_pos_left, nr_pos_right)

# --------------------------------------------------

collision_checker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[arm_left_index, arm_right_index],
    edge_step_size=0.01,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
)

# --------------------------------------------------

q = plant.GetPositions(plant_context, arm_left_index).tolist()
pyout(q)

# --------------------------------------------------

q_all = plant.GetPositions(plant_context)
pyout(collision_checker.CheckConfigCollisionFree(q_all))

# --------------------------------------------------

q_left = q.copy()
q_right = q.copy()
q_left[1] = -np.pi / 2
q_right[1] = -np.pi / 2
q_right[2] = np.pi / 2
plant.SetPositions(plant_context, arm_left_index, q_left)
plant.SetPositions(plant_context, arm_right_index, q_right)

diagram.ForcedPublish(context)

q_all = plant.GetPositions(plant_context)
pyout(collision_checker.CheckConfigCollisionFree(q_all))

# --------------------------------------------------

pyout(collision_checker.CheckConfigCollisionFree(np.zeros(12)))


pyout()
time.sleep(600)
