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


class CalibrateSophieRS2(Subprocedure):
    SCENE = "wilson_holds_charuco"
    N_MEASUREMENTS = 10
    BOARD_Z = 0.7
    DISTANCE = 0.3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rs2: RS2Client = RS2Client()
        self.data_root = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )

    def run(self):
        self.__wilson_display()
        err, err_ = np.inf, np.inf
        measurements = []
        while len(measurements) < self.N_MEASUREMENTS or np.isinf(err) or err_ < err:
            err = err_
            dxyz = np.array(
                [
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, -0.5),
                    np.random.uniform(-0.3, 0.3),
                ]
            )

            dxyz = dxyz / np.linalg.norm(dxyz) * self.DISTANCE
            xyz_coords = np.array([0.0, 0.0, self.BOARD_Z]) + dxyz
            tcp = horizontal_view_rotation_matrix(z_axis=-dxyz / np.linalg.norm(dxyz))
            tcp[:3, 3] = xyz_coords

            R = homogenous_transformation(
                roll=np.random.uniform(-10, 10),
                pitch=np.random.uniform(-10, 10),
                yaw=np.random.uniform(-179, 179),
            )
            tcp = tcp @ R

            try:
                plan = self.ompl.plan_to_tcp_pose(sophie=tcp, scene=self.SCENE)
                self.sophie.execute_plan(plan)
                time.sleep(1)
            except RuntimeError:
                continue

            try:
                measurements.append(
                    TCPs(
                        tcp_cam=CameraCalibration.wait_for_measurements(
                            self.rs2, nr_of_frames=16
                        ),
                        tcp_arm=self.sophie.get_tcp_pose(),
                    )
                )

                _, err_ = CameraCalibration.compute_calibration_from_measurements(
                    measurements, mode="eye_in_hand"
                )
                ros.loginfo(f"RS2_Sophie error: {err_} ({len(measurements)})")
            except TimeoutError:
                continue

        poses, errors = CameraCalibration.compute_calibration_from_measurements(
            measurements, mode="eye_in_hand"
        )
        ros.loginfo(f"RS2_Sophie error: {errors}")
        save_path = self.data_root / "T_rs2_tcp_sophie.npy"
        np.save(save_path, poses)

    def __wilson_display(self):
        board = CharucoBoard()
        tcp = np.array(
            [
                [0.0, 0.0, 1.0, -board.width / 2],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, self.BOARD_Z + board.height / 2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_tcp_pose(
            wilson=tcp,
            scene=self.SCENE,
        )
        self.wilson.execute_plan(plan)
