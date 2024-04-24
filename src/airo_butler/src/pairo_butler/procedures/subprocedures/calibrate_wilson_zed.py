import numpy as np
from pairo_butler.utils.transformations_3d import homogenous_transformation
from pairo_butler.camera.calibration import CameraCalibration, TCPs
from pairo_butler.procedures.subprocedure import Subprocedure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rospkg
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros


class CalibrateWilsonZed(Subprocedure):
    SCENE = "wilson_holds_charuco"
    N_MEASUREMENTS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.zed: ZEDClient = ZEDClient()
        self.data_root = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )

    def run(self):
        measurements = []

        while len(measurements) < self.N_MEASUREMENTS:
            tcp = self.config.tcp_calibration_wilson
            H = homogenous_transformation(
                roll=np.random.uniform(-179, 179),
                pitch=np.random.uniform(-10, 10),
                yaw=np.random.uniform(-10, 10),
                dx=np.random.uniform(-0.1, 0.1),
                dy=np.random.uniform(-0.2, 0.2),
                dz=np.random.uniform(-0.2, 0.2),
            )

            try:
                plan = self.ompl.plan_to_tcp_pose(wilson=tcp @ H, scene=self.SCENE)
                self.wilson.execute_plan(plan)
            except RuntimeError:
                continue

            try:
                measurements.append(
                    TCPs(
                        tcp_cam=CameraCalibration.wait_for_measurements(
                            self.zed,
                        ),
                        tcp_arm=self.wilson.get_tcp_pose(),
                    )
                )
            except TimeoutError:
                continue

        poses, errors = CameraCalibration.compute_calibration_from_measurements(
            measurements, mode="eye_to_hand"
        )
        ros.loginfo(f"Zed_Wilson error: {errors}")
        save_path = self.data_root / "T_zed_wilson.npy"
        np.save(save_path, poses)
