from functools import partial
import time
import airo_models
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from munch import Munch
from pairo_butler.motion_planning.custom_constraints import (
    DistanceBetweenToolsConstraint,
)
from pairo_butler.motion_planning.obstacles import CharucoBoard, HangingTowel
from pairo_butler.motion_planning.towel_obstacle import TowelObstacle
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
import rospy as ros
from typing import Any, Callable, List, Optional, Tuple
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
from cloth_tools.path.execution import time_parametrize_toppra
from ur_analytic_ik import ur5e
from pydrake.trajectories import Trajectory
from pydrake.multibody.plant import MultibodyPlant


np.set_printoptions(precision=2, suppress=True)


class DrakeSimulation:
    def __init__(
        self,
        scene_name: str = "default",
        min_distance: float | None = None,
        max_distance: float | None = None,
    ):
        self.config = load_config()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Init subscribers
        self.sophie: UR5eClient = UR5eClient("sophie")
        self.wilson: UR5eClient = UR5eClient("wilson")

        self.tcp_transform: np.ndarray = self.__initialize_tcp_transform()
        self.robot_diagram_builder: RobotDiagramBuilder
        self.is_state_valid_fn: Callable[..., bool]
        self.sophie_idx: Any
        self.wilson_idx: Any
        self.object_idx: Any = None
        self.diagram: Any
        self.context: Any
        self.plant: MultibodyPlant
        self.plant_context: Any
        self.meshcat: Any

        self.__create_scene(scene_name, min_distance, max_distance)

        self.planner: DualArmOmplPlanner
        self.__create_planner()

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

    def reset(self, towel: Optional[TowelObstacle]):
        self.__create_scene(towel)
        self.__create_planner()

    def update_towel(self, towel: Optional[TowelObstacle] = None):
        wilson_tcp = self.wilson.get_tcp_pose()

        body = self.diagram.plant().GetBodyByName("base_link", self.object_idx)
        self.diagram.plant().SetFreeBodyPose(
            self.diagram.plant().GetMyContextFromRoot(self.context),
            body,
            RigidTransform(p=wilson_tcp[:3, 3].tolist() - np.array([0.0, 0.0, 0.1])),
        )

        pyout(type(self.plant))

    def __initialize_tcp_transform(self):
        tcp_transform = np.identity(4)
        tcp_transform[2, 3] = self.config.gripper_length
        return tcp_transform

    def __create_scene(
        self,
        scenario: str = "default",
        min_distance: float | None = None,
        max_distance: float | None = None,
    ):
        self.robot_diagram_builder = RobotDiagramBuilder()
        self.meshcat = add_meshcat_to_builder(self.robot_diagram_builder)

        arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(
            self.robot_diagram_builder
        )
        plant = self.robot_diagram_builder.plant()
        parser = self.robot_diagram_builder.parser()

        # Add front wall to table

        wall_length = 1.5
        wall_front_urdf_path = airo_models.box_urdf_path(
            (0.2, wall_length, 2.0), "wall_front"
        )
        wall_front_index = parser.AddModels(wall_front_urdf_path)[0]
        wall_front_frame = plant.GetFrameByName("base_link", wall_front_index)
        wall_front_transform = RigidTransform(p=[-0.8, wall_length / 2 - 0.15, 0])
        world_frame = plant.world_frame()
        plant.WeldFrames(world_frame, wall_front_frame, wall_front_transform)

        self.wilson_idx, self.sophie_idx = arm_indices

        realsense_urdf_path = airo_models.get_urdf_path("d435")

        rs2_indexes = []
        for arm_index in (self.wilson_idx, self.sophie_idx):
            arm_tool_frame = plant.GetFrameByName("tool0", arm_index)

            realsense_index = parser.AddModels(realsense_urdf_path)[0]
            rs2_indexes.append(realsense_index)
            realsense_frame = plant.GetFrameByName("base_link", realsense_index)

            X_Tool0_RealsenseBase = RigidTransform(
                rpy=RollPitchYaw([0, 0, 0]), p=[0, -0.06, 0]
            )

            plant.WeldFrames(arm_tool_frame, realsense_frame, X_Tool0_RealsenseBase)

        # Add obstacle
        if scenario == "wilson_holds_charuco" or scenario == "sophie_holds_charuco":
            arm_idx = (
                self.wilson_idx
                if scenario == "wilson_holds_charuco"
                else self.sophie_idx
            )

            board = CharucoBoard()
            arm_tool_frame = plant.GetFrameByName("tool0", arm_idx)

            self.object_idx = parser.AddModels(board.urdf)[0]
            object_frame = plant.GetFrameByName("base_link", self.object_idx)

            p = [0, board.height / 2, 0.165 + board.width / 2]
            rpy = np.deg2rad([0, 90, 0])

            object_transform = RigidTransform(rpy=RollPitchYaw(rpy), p=p)
            plant.WeldFrames(arm_tool_frame, object_frame, object_transform)
        elif scenario == "hanging_towel":
            towel = HangingTowel()
            self.object_idx = parser.AddModels(towel.urdf)[0]
            object_frame = plant.GetFrameByName("base_link", self.object_idx)
            world_frame = plant.world_frame()
            p = [0, 0, towel.bottom + towel.length / 2]
            plant.WeldFrames(
                world_frame,
                object_frame,
                RigidTransform(rpy=RollPitchYaw(np.deg2rad([0, 0, 45])), p=p),
            )

        self.diagram, self.context = finish_build(
            self.robot_diagram_builder, self.meshcat
        )

        collision_indexes = [*arm_indices, *gripper_indices, *rs2_indexes]
        if self.object_idx is not None:
            collision_indexes.append(self.object_idx)

        self.collision_checker = SceneGraphCollisionChecker(
            model=self.diagram,
            robot_model_instances=collision_indexes,
            edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
            env_collision_padding=0.005,
            self_collision_padding=0.005,
        )

        if max_distance is None and min_distance is None:
            self.is_state_valid_fn = self.collision_checker.CheckConfigCollisionFree
        else:
            min_distance = 0.05 if min_distance is None else min_distance
            max_distance = 999.0 if max_distance is None else max_distance

            self.is_state_valid_fn = DistanceBetweenToolsConstraint(
                self.collision_checker.CheckConfigCollisionFree,
                min_distance,
                max_distance,
                tcp_transform=self.config.tcp_transform,
            )

        self.plant = self.diagram.plant()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        transform_0 = RigidTransform(p=[0, 0, 0.35], rpy=RollPitchYaw([np.pi, 0, 0]))

        add_meshcat_triad(self.meshcat, "TCP Frame left", X_W_Triad=transform_0)

        transform_1 = RigidTransform(
            p=[0.15, 0, 0.3], rpy=RollPitchYaw([np.pi / 2, 0, np.pi / 2])
        )

        add_meshcat_triad(self.meshcat, "TCP Frame right", X_W_Triad=transform_1)

    def __create_planner(self):
        def inverse_kinematics_in_world_fn(
            tcp_pose: HomogeneousMatrixType, X_W_CB: HomogeneousMatrixType
        ) -> List[JointConfigurationType]:
            X_W_TCP = tcp_pose  # pose_tcp_in_worldframe
            X_CB_W = np.linalg.inv(X_W_CB)  # pose world in controlbox
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
            max_planning_time=5.0 if self.max_distance is None else 10.0,
            num_interpolated_states=self.config.num_interpolated_states,
        )

    def plan_to_tcp_pose(
        self,
        sophie_tgt: Optional[np.ndarray],
        wilson_tgt: Optional[np.ndarray],
        sophie_start: np.ndarray | None = None,
        wilson_start: np.ndarray | None = None,
        desirable_goal_configurations_sophie: List[np.ndarray] | None = None,
        desirable_goal_configurations_wilson: List[np.ndarray] | None = None,
    ):
        if wilson_start is None:
            wilson_start = np.clip(
                self.wilson.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )
        if sophie_start is None:
            sophie_start = np.clip(
                self.sophie.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )

        if desirable_goal_configurations_sophie is None:
            desirable_goal_configurations_sophie = [
                self.sophie.get_joint_configuration()
            ]
        if desirable_goal_configurations_wilson is None:
            desirable_goal_configurations_wilson = [
                self.wilson.get_joint_configuration()
            ]

        path = self.planner.plan_to_tcp_pose(
            wilson_start,
            sophie_start,
            wilson_tgt,
            sophie_tgt,
            desirable_goal_configurations_left=desirable_goal_configurations_wilson,
            desirable_goal_configurations_right=desirable_goal_configurations_sophie,
        )

        joint_trajectory, time_trajectory = time_parametrize_toppra(path, self.plant)

        period = 0.005
        duration = time_trajectory.end_time()
        n_servos = int(np.ceil(duration / period))
        period_adjusted = duration / n_servos

        path_sophie = []
        path_wilson = []
        for t in np.linspace(0, duration, n_servos):
            joints = joint_trajectory.value(time_trajectory.value(t).item()).squeeze()
            path_wilson.append(joints[0:6])
            path_sophie.append(joints[6:12])

        path_sophie = np.stack(path_sophie, axis=0)
        path_wilson = np.stack(path_wilson, axis=0)

        return path_sophie, path_wilson, period_adjusted

    def plan_to_joint_configuration(
        self,
        sophie_tgt: Optional[np.ndarray],
        wilson_tgt: Optional[np.ndarray],
        sophie_start: np.ndarray | None = None,
        wilson_start: np.ndarray | None = None,
        max_distance: float | None = None,
    ):
        if wilson_start is None:
            wilson_start = np.clip(
                self.wilson.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )
        if sophie_start is None:
            sophie_start = np.clip(
                self.sophie.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )

        path = self.planner.plan_to_joint_configuration(
            wilson_start, sophie_start, wilson_tgt, sophie_tgt
        )

        joint_trajectory, time_trajectory = time_parametrize_toppra(path, self.plant)

        period = 0.005
        duration = time_trajectory.end_time()
        n_servos = int(np.ceil(duration / period))
        period_adjusted = duration / n_servos

        path_sophie = []
        path_wilson = []
        for t in np.linspace(0, duration, n_servos):
            joints = joint_trajectory.value(time_trajectory.value(t).item()).squeeze()
            path_wilson.append(joints[0:6])
            path_sophie.append(joints[6:12])

        path_sophie = np.stack(path_sophie, axis=0)
        path_wilson = np.stack(path_wilson, axis=0)

        return path_sophie, path_wilson, period_adjusted

    def toppra(
        self,
        sophie_path: np.ndarray | None = None,
        wilson_path: np.ndarray | None = None,
    ):
        if sophie_path is None and wilson_path is None:
            raise RuntimeError(
                f"At least one of sophie_path and wilson_path must be not None"
            )

        if wilson_path is None:
            wilson_path = np.stack(
                [
                    np.clip(
                        self.wilson.get_joint_configuration(),
                        a_min=self.config.joint_bounds_lower,
                        a_max=self.config.joint_bounds_upper,
                    )
                ]
                * sophie_path.shape[0],
                axis=0,
            )
        if sophie_path is None:
            sophie_path = np.stack(
                [
                    np.clip(
                        self.sophie.get_joint_configuration(),
                        a_min=self.config.joint_bounds_lower,
                        a_max=self.config.joint_bounds_upper,
                    )
                ]
                * wilson_path.shape[0],
                axis=0,
            )

        path = np.concatenate((wilson_path, sophie_path), axis=1)
        joint_trajectory, time_trajectory = time_parametrize_toppra(path, self.plant)

        period = 0.005
        duration = time_trajectory.end_time()
        n_servos = int(np.ceil(duration / period))
        period_adjusted = duration / n_servos

        path_sophie = []
        path_wilson = []
        for t in np.linspace(0, duration, n_servos):
            joints = joint_trajectory.value(time_trajectory.value(t).item()).squeeze()
            path_wilson.append(joints[0:6])
            path_sophie.append(joints[6:12])

        path_sophie = np.stack(path_sophie, axis=0)
        path_wilson = np.stack(path_wilson, axis=0)

        return path_sophie, path_wilson, period_adjusted

    def get_ik_solutions(
        self,
        sophie_tgt: Optional[np.ndarray],
        wilson_tgt: Optional[np.ndarray],
        sophie_start: np.ndarray | None = None,
        wilson_start: np.ndarray | None = None,
    ):
        if wilson_start is None:
            wilson_start = np.clip(
                self.wilson.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )
        if sophie_start is None:
            sophie_start = np.clip(
                self.sophie.get_joint_configuration(),
                a_min=self.config.joint_bounds_lower,
                a_max=self.config.joint_bounds_upper,
            )

        solutions = self.planner.get_ik_solutions(
            wilson_start,
            sophie_start,
            wilson_tgt,
            sophie_tgt,
        )

        poses_sophie = []
        poses_wilson = []
        for solution in solutions:
            poses_wilson.append(solution[0:6])
            poses_sophie.append(solution[6:12])

        poses_sophie = np.stack(poses_sophie, axis=0)
        poses_wilson = np.stack(poses_wilson, axis=0)

        return poses_sophie, poses_wilson
