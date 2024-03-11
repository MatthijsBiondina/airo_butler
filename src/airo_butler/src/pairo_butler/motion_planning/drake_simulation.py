from functools import partial
import time
import airo_models
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
import rospy as ros
from typing import Any, Callable, List, Optional
import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pairo_butler.utils.tools import load_config, pyout
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.drake.visualization import (
    publish_dual_arm_joint_path,
    publish_ik_solutions,
    add_meshcat_triad,
)
from cloth_tools.drake.scenes import X_W_L_DEFAULT, X_W_R_DEFAULT, X_CB_B
from ur_analytic_ik import ur5e


class DrakeSimulation:
    def __init__(self):
        self.config = load_config()

        self.tcp_transform: np.ndarray = self.__initialize_tcp_transform()
        self.robot_diagram_builder: RobotDiagramBuilder
        self.is_state_valid_fn: Callable[..., bool]
        self.sophie_idx: Any
        self.wilson_idx: Any
        self.diagram: Any
        self.context: Any
        self.plant: Any
        self.plant_context: Any
        self.__create_default_scene()

        self.planner: DualArmOmplPlanner
        self.__create_planner()

        # Init subscribers
        self.sophie: UR5eClient = UR5eClient("sophie")
        self.wilson: UR5eClient = UR5eClient("wilson")

    def update(self):
        try:
            self.plant.SetPositions(
                self.plant_context,
                self.sophie_idx,
                self.sophie.get_joint_configuration(),
            )
            self.plant.SetPositions(
                self.plant_context,
                self.wilson_idx,
                self.wilson.get_joint_configuration(),
            )
            self.diagram.ForcedPublish(self.context)
        except AttributeError:
            pass

    def __initialize_tcp_transform(self):
        tcp_transform = np.identity(4)
        tcp_transform[2, 3] = self.config.gripper_length
        return tcp_transform

    def __create_default_scene(self):
        self.robot_diagram_builder = RobotDiagramBuilder()
        self.meshcat = add_meshcat_to_builder(self.robot_diagram_builder)
        arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(
            self.robot_diagram_builder
        )

        self.wilson_idx, self.sophie_idx = arm_indices

        # plant = self.robot_diagram_builder.plant()
        # parser = self.robot_diagram_builder.parser()
        # realsense_urdf_path = airo_models.get_urdf_path("d435")

        # for arm_index in (self.wilson_idx, self.sophie_idx):
        #     arm_tool_frame = plant.GetFrameByName("tool0", arm_index)

        #     realsense_index = parser.AddModels(realsense_urdf_path)[0]
        #     realsense_frame = plant.GetFrameByName("base_link", realsense_index)

        #     X_Tool0_RealsenseBase = RigidTransform(
        #         rpy=RollPitchYaw([0, 0, 0]), p=[0, -0.06, 0]
        #     )

        #     plant.WeldFrames(arm_tool_frame, realsense_frame, X_Tool0_RealsenseBase)

        self.diagram, self.context = finish_build(
            self.robot_diagram_builder, self.meshcat
        )

        collision_checker = SceneGraphCollisionChecker(
            model=self.diagram,
            robot_model_instances=[*arm_indices, *gripper_indices],
            edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
            env_collision_padding=0.005,
            self_collision_padding=0.005,
        )

        self.is_state_valid_fn = collision_checker.CheckConfigCollisionFree

        self.plant = self.diagram.plant()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        transform_0 = RigidTransform(p=[0, 0, 0.35], rpy=RollPitchYaw([np.pi, 0, 0]))
        tcp_pose_0 = np.ascontiguousarray(transform_0.GetAsMatrix4())

        add_meshcat_triad(self.meshcat, "TCP Frame left", X_W_Triad=transform_0)

        transform_1 = RigidTransform(
            p=[0.15, 0, 0.3], rpy=RollPitchYaw([np.pi / 2, 0, np.pi / 2])
        )
        tcp_pose_1 = np.ascontiguousarray(transform_1.GetAsMatrix4())

        add_meshcat_triad(self.meshcat, "TCP Frame right", X_W_Triad=transform_1)

    def __create_planner(self):
        def inverse_kinematics_in_world_fn(
            tcp_pose: HomogeneousMatrixType, X_W_CB: HomogeneousMatrixType
        ) -> List[JointConfigurationType]:
            X_W_TCP = tcp_pose
            X_CB_W = np.linalg.inv(X_W_CB)
            solutions_1x6 = ur5e.inverse_kinematics_with_tcp(
                X_CB_W @ X_W_TCP, np.array(self.config.tcp_transform)
            )
            solutions = [solution.squeeze() for solution in solutions_1x6]
            return solutions

        inverse_kinematics_wilson = partial(
            inverse_kinematics_in_world_fn,
            X_W_CB=(X_W_L_DEFAULT @ X_CB_B.inverse()).GetAsMatrix4(),
        )
        inverse_kinematics_sophie = partial(
            inverse_kinematics_in_world_fn,
            X_W_CB=(X_W_R_DEFAULT @ X_CB_B.inverse()).GetAsMatrix4(),
        )

        joint_bounds = (
            np.deg2rad(self.config.joint_bounds_lower),
            np.deg2rad(self.config.joint_bounds_upper),
        )

        self.planner = DualArmOmplPlanner(
            self.is_state_valid_fn,
            inverse_kinematics_wilson,
            inverse_kinematics_sophie,
            joint_bounds_left=joint_bounds,
            joint_bounds_right=joint_bounds,
        )

    def plan_to_tcp_pose(
        self, sophie_tgt: Optional[np.ndarray], wilson_tgt: Optional[np.ndarray]
    ):
        wilson_start = self.wilson.get_joint_configuration()
        sophie_start = self.sophie.get_joint_configuration()

        path = self.planner.plan_to_tcp_pose(
            wilson_start, sophie_start, wilson_tgt, sophie_tgt
        )

        wilson_path = np.stack([state[0] for state in path], axis=0)
        sophie_path = np.stack([state[1] for state in path], axis=1)

        pyout()
