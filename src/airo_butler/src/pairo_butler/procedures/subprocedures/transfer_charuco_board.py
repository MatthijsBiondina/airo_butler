import time
import numpy as np
from pairo_butler.utils.transformations_3d import (
    homogenous_transformation,
    horizontal_view_rotation_matrix,
)
from pairo_butler.utils.tools import pyout
from pairo_butler.motion_planning.obstacles import CharucoBoard
from pairo_butler.camera.rs2_camera import RS2Client
from pairo_butler.camera.calibration import CameraCalibration, TCPs
from pairo_butler.procedures.subprocedure import Subprocedure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rospkg
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros

from pairo_butler.procedures.subprocedure import Subprocedure


class TransferCharucoBoard(Subprocedure):
    SCENE = "wilson_holds_charuco"
    BOARD_Z = 0.7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rs2 = RS2Client()

        self.data_root = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )
        self.T_rs2_sophie = np.load(self.data_root / "T_rs2_tcp_sophie.npy")

    def run(self):
        tcp_charuco = self.__watch_board_pose()
        tcp_approach, tcp_grasp = self.__calculate_grasp_poses(tcp_charuco)

        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_approach, scene=self.SCENE)
        self.sophie.execute_plan(plan)

        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_grasp, scene="default")
        self.sophie.execute_plan(plan)

        self.sophie.close_gripper()
        ros.sleep(1)
        self.wilson.open_gripper()

        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_approach, scene="default")
        self.sophie.execute_plan(plan)

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie),
            wilson=np.deg2rad(self.config.joints_rest_wilson),
            scene="sophie_holds_charuco",
        )
        self.sophie.execute_plan(plan)

    def __watch_board_pose(self):
        board = CharucoBoard()
        tcp_wilson = np.array(
            [
                [-1.0, 0.0, 0.0, 0.35],
                [0.0, 0.0, -1.0, board.width / 2 - board.square_size],
                [0.0, -1.0, 0.0, self.BOARD_Z + board.height / 2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        tcp_sophie = np.array(
            [
                [0.0, 0.0, 1.0, -0.1],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, self.BOARD_Z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        try:
            plan = self.ompl.plan_to_tcp_pose(wilson=tcp_wilson, scene=self.SCENE)
            self.wilson.execute_plan(plan)
            plan = self.ompl.plan_to_tcp_pose(sophie=tcp_sophie, scene=self.SCENE)
            self.sophie.execute_plan(plan)
        except RuntimeError:
            plan = self.ompl.plan_to_tcp_pose(
                wilson=tcp_wilson, sophie=tcp_sophie, scene=self.SCENE
            )
            self.wilson.execute_plan(plan)

        charuco_sophie = CameraCalibration.wait_for_measurements(self.rs2)
        charuco_world = self.sophie.get_tcp_pose() @ self.T_rs2_sophie @ charuco_sophie

        return charuco_world

    def __calculate_grasp_poses(self, tcp_charuco: np.ndarray):
        board = CharucoBoard()

        tcp_approach = self.wilson.get_tcp_pose()
        tcp_approach[:3, 3] += (board.width + board.square_size) * tcp_approach[:3, 2]
        tcp_approach = tcp_approach @ homogenous_transformation(pitch=180)

        tcp_grasp = np.eye(4)
        tcp_grasp[:3, 0] = tcp_charuco[:3, 2]
        tcp_grasp[:3, 1] = tcp_charuco[:3, 1]
        tcp_grasp[:3, 2] = -tcp_charuco[:3, 0]
        tcp_grasp[:3, 3] = (
            tcp_charuco[:3, 3]
            + (board.width - board.square_size - 0.01) * tcp_charuco[:3, 0]
        )

        return tcp_approach, tcp_grasp
