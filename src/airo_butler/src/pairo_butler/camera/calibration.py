from copy import deepcopy
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2

import numpy as np
import rospkg
import yaml
from pairo_butler.camera.zed_camera import ZEDClient
from pairo_butler.camera.rs2_camera import RS2Client
from pairo_butler.ur3_arms.ur3_constants import WILSON_SLEEP, SOPHIE_SLEEP
from pairo_butler.utils.tools import pbar, pyout
from pairo_butler.ur3_arms.ur3_client import UR3Client
from pairo_butler.utils.pods import ZEDPOD, ImagePOD
import rospy as ros
from airo_butler.msg import PODMessage
import scipy.spatial.transform as transform
from airo_camera_toolkit.calibration.collect_calibration_data import (
    create_calibration_data_dir,
    save_calibration_sample,
)
from airo_camera_toolkit.calibration.compute_calibration import (
    compute_calibration_all_methods,
    compute_calibration,
)
from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_charuco_board,
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
)

# Arm poses

SOPHIE_SLEEP = np.array([+0.00, -0.75, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_SLEEP = np.array([+0.00, -0.25, -0.50, -0.50, +0.50, +0.00]) * np.pi

APPROACH_DISTANCE = 0.1

# States
STATE_STARTUP = 0
STATE_GRAB_CHARUCO_BOARD = 1
STATE_CALIBRATE_ZED_WILSON = 2
STATE_CALIBRATE_RS2_SOPHIE = 3
STATE_TRANSFER_CHARUCO_BOARD = 4
STATE_CALIBRATE_ZED_SOPHIE = 5
STATE_DROP_CHARUCO_BOARD = 6
STATE_DONE = 7


cv2_CALIBRATION_METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Haraud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


np.set_printoptions(suppress=True, precision=3)


class TCPs:
    __slots__ = ["cam", "arm"]

    def __init__(self, tcp_cam: np.ndarray, tcp_arm: np.ndarray) -> None:
        for tcp in (tcp_cam, tcp_arm):
            assert tcp.shape == (4, 4), "TCP pose must be a 4x4 matrix."
            assert np.allclose(
                tcp[3, :], np.array([0, 0, 0, 1])
            ), "The last row of TCP pose must be [0, 0, 0, 1]."
            R = tcp[:3, :3]
            assert np.allclose(
                np.dot(R, R.T), np.identity(3), atol=1e-6
            ), "The rational component of TCP pose should be orthogonal."
            assert np.isclose(
                np.linalg.det(R), 1, atol=1e-6
            ), "The rational component of TCP pose should have determinant one."

        self.cam = tcp_cam
        self.arm = tcp_arm


class CameraCalibration:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2
    CHARUCO_FRAMES = 16

    """
    State machine for calibrating eye-to-hand zed2i camera
    """

    def __init__(self, name: str = "zed_calibrator") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        self.zed: Optional[ZEDClient] = None
        self.rs2: Optional[RS2Client] = None
        self.wilson: Optional[UR3Client] = None
        self.sophie: Optional[UR3Client] = None

        # Placeholders:
        self.state: str = STATE_STARTUP
        # self.state: str = STATE_CALIBRATE_ZED_SOPHIE

        # Filesystem
        self.data_root = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )
        with open(
            Path(sys.modules[self.__class__.__module__].__file__).parent
            / "calibration_poses.yaml",
            "r",
        ) as f:
            self.poses = yaml.safe_load(f)
            for name_, pose_in_deg in self.poses.items():
                self.poses[name_] = np.deg2rad(pose_in_deg)

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.zed = ZEDClient()
        self.rs2 = RS2Client()
        self.wilson = UR3Client("wilson")
        self.sophie = UR3Client("sophie")

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.state == STATE_STARTUP:
                self.state = self.__startup()
            if self.state == STATE_GRAB_CHARUCO_BOARD:
                self.state = self.__grab()
            elif self.state == STATE_CALIBRATE_ZED_WILSON:
                self.state = self.__calibrate_wilson()
            elif self.state == STATE_CALIBRATE_RS2_SOPHIE:
                self.state = self.__calibrate_sophie_rs2()
            elif self.state == STATE_TRANSFER_CHARUCO_BOARD:
                self.state = self.__transfer_charuco_board()
            elif self.state == STATE_CALIBRATE_ZED_SOPHIE:
                self.state = self.__calibrate_sophie_zed()
            elif self.state == STATE_DROP_CHARUCO_BOARD:
                self.state = self.__drop_charuco_board()
            elif self.state == STATE_DONE:
                ros.signal_shutdown("Done with calibration!")
            else:
                ros.logwarn(f"Unknown state: {self.state}")
            self.rate.sleep()

    def __startup(self):
        self.sophie.move_to_joint_configuration(self.poses["sophie_rest"])
        self.wilson.move_to_joint_configuration(self.poses["wilson_rest"])
        self.sophie.open_gripper()
        self.wilson.close_gripper()

        return STATE_GRAB_CHARUCO_BOARD  # <- correct

    def __grab(self):
        ros.loginfo(f"Give the Charuco board to Wilson.")
        ros.sleep(1.0)
        while not ros.is_shutdown() and self.wilson.get_gripper_width() < 0.002:
            self.wilson.open_gripper()
            self.wilson.close_gripper()
            ros.sleep(1.0)
        ros.loginfo(f"Good human!")
        return STATE_CALIBRATE_ZED_WILSON

    def __calibrate_wilson(self):
        measurements: List[TCPs] = []
        for joint_config in pbar(
            self.poses["wilson_calibration_poses"], desc="ZED -> Wilson"
        ):
            self.wilson.move_to_joint_configuration(joint_config)

            try:
                measurements.append(
                    TCPs(
                        tcp_cam=self.__wait_for_measurements(ros.Time.now()),
                        tcp_arm=self.wilson.get_tcp_pose(),
                    )
                )
            except TimeoutError:
                ros.logwarn(f"Cannot see the charuco board.")

        zed_pose, error = self.__compute_calibration(measurements, mode="eye_to_hand")
        ros.loginfo(f"Calibrated Wilson with error: {error:.3f}")
        save_path = self.data_root / "T_zed_wilson.npy"
        np.save(save_path, zed_pose)

        self.wilson.move_to_joint_configuration(self.poses["wilson_rest"])

        return STATE_CALIBRATE_RS2_SOPHIE

    def __calibrate_sophie_zed(self):
        measurements: List[TCPs] = []
        for joint_config in pbar(
            self.poses["sophie_eye_to_hand_poses"], desc="ZED -> Sophie"
        ):
            self.sophie.move_to_joint_configuration(joint_config)

            try:
                measurements.append(
                    TCPs(
                        tcp_cam=self.__wait_for_measurements(ros.Time.now()),
                        tcp_arm=self.sophie.get_tcp_pose(),
                    )
                )
            except TimeoutError:
                ros.logwarn(f"Cannot see the charuco board.")

        zed_pose, error = self.__compute_calibration(measurements, mode="eye_to_hand")
        ros.loginfo(f"Calibrated Sophie ZED with error: {error:.3f}")
        save_path = self.data_root / "T_zed_sophie.npy"
        np.save(save_path, zed_pose)

        self.sophie.move_to_joint_configuration(self.poses["sophie_rest"])

        return STATE_DROP_CHARUCO_BOARD

    def __calibrate_sophie_rs2(self):
        self.wilson.move_to_joint_configuration(self.poses["wilson_present_to_sophie"])

        measurements: List[TCPs] = []
        for joint_config in pbar(
            self.poses["sophie_eye_in_hand_poses"], desc="RS2 -> Sophie"
        ):
            self.sophie.move_to_joint_configuration(joint_config)
            measurements.append(
                TCPs(
                    tcp_cam=self.__wait_for_measurements(ros.Time.now(), camera="rs2"),
                    tcp_arm=self.sophie.get_tcp_pose(),
                )
            )
        rs2_pose, error = self.__compute_calibration(measurements, mode="eye_in_hand")
        ros.loginfo(f"Calibrated Sophie rs2 with error: {error:.3f}")
        save_path = self.data_root / "T_rs2_tcp_sophie.npy"
        np.save(save_path, rs2_pose)

        self.wilson.move_to_joint_configuration(self.poses["wilson_rest"])
        self.sophie.move_to_joint_configuration(self.poses["sophie_rest"])

        return STATE_TRANSFER_CHARUCO_BOARD

    def __transfer_charuco_board(self):
        self.wilson.move_to_joint_configuration(self.poses["wilson_transfer_pose"])
        self.sophie.move_to_joint_configuration(
            self.poses["sophie_look_before_transfer"]
        )
        self.sophie.open_gripper()

        # location of board relative to wilson
        # T_charuco_zed = self.__wait_for_measurements(ros.Time.now(), camera="zed")
        # T_zed_wilson = np.load(self.data_root / "T_zed_wilson.npy")
        # T_charuco_wilson = T_zed_wilson @ T_charuco_zed
        # pyout(f"T_charuco_wilson:\n{T_charuco_wilson}")
        # pyout(f"Wilson TCP:\n{self.wilson.get_tcp_pose()}")

        # location of board relative to sophie
        T_charuco_rs2 = self.__wait_for_measurements(ros.Time.now(), camera="rs2")
        T_rs2_tcp_sophie = np.load(self.data_root / "T_rs2_tcp_sophie.npy")
        T_tcp_sophie = self.sophie.get_tcp_pose()
        T_charuco_sophie = T_tcp_sophie @ T_rs2_tcp_sophie @ T_charuco_rs2
        pyout(f"T_charuco_sophie:\n{T_charuco_sophie}")
        pyout(f"Sophie TCP:\n{T_tcp_sophie}")

        # Compute grasp pose for sophie:
        charuco_board_nr_of_squares_along_width = (
            AIRO_DEFAULT_CHARUCO_BOARD.getChessboardSize()[0]
        )
        charuco_board_square_size = AIRO_DEFAULT_CHARUCO_BOARD.getSquareLength()
        grasp_X = (
            T_charuco_sophie[:3, 3]
            + T_charuco_sophie[:3, 0]
            * (charuco_board_nr_of_squares_along_width - 0.75)
            * charuco_board_square_size
        )
        grasp_z = -T_charuco_sophie[:3, 0]
        grasp_z[-1] = 0.0
        grasp_z /= np.linalg.norm(grasp_z)

        self.sophie.grasp_horizontal(world_pos=grasp_X, gripper_z_dir=grasp_z)
        self.wilson.open_gripper()
        self.wilson.move_to_joint_configuration(self.poses["wilson_rest"])
        self.sophie.move_to_joint_configuration(self.poses["sophie_rest"])

        return STATE_CALIBRATE_ZED_SOPHIE

    def __drop_charuco_board(self):
        initial_world_pos = np.array([-0.50, -0.3, 0.40])
        self.sophie.move_to_tcp_vertical_down(initial_world_pos)
        self.sophie.move_to_tcp_vertical_down(np.array([-0.50, -0.3, 0.27]))
        self.sophie.move_gripper(0.02, blocking=True)
        self.sophie.move_to_tcp_vertical_down(np.array([-0.50, +0.2, 0.27]))
        self.sophie.open_gripper()
        self.sophie.move_to_joint_configuration(self.poses["sophie_rest"])

        return STATE_DONE

    def __wait_for_measurements(self, t0: ros.Time, timeout=10, camera: str = "zed"):
        t_start = ros.Time.now()
        timestamps: List[ros.Time] = []
        imgs: List[np.ndarray] = []
        TCP: List[np.ndarray] = []

        while not ros.is_shutdown() and len(imgs) < self.CHARUCO_FRAMES:
            if ros.Time.now() > t_start + ros.Duration(timeout):
                raise TimeoutError
            if camera == "zed":
                pod = deepcopy(self.zed.pod)
                img = (pod.rgb_image * 255)[..., ::-1].astype(np.uint8)
            elif camera == "rs2":
                pod = deepcopy(self.rs2.pod)
                img = np.array(pod.image)

            if pod is None:
                pass
            elif pod.timestamp in timestamps:
                pass
            elif pod.timestamp < t0:
                pass
            else:
                tcp = detect_charuco_board(
                    img,
                    pod.intrinsics_matrix,
                    aruco_dict=AIRO_DEFAULT_ARUCO_DICT,
                    charuco_board=AIRO_DEFAULT_CHARUCO_BOARD,
                )
                if tcp is not None:
                    TCP.append(tcp)
                    imgs.append(img)
                    timestamps.append(pod.timestamp)

            self.rate.sleep()

        # Average transforms:
        M = np.stack(TCP, axis=0)
        tr = np.mean(M[:, :3, 3], axis=0)
        Q = np.stack([transform.Rotation.from_matrix(m[:3, :3]).as_quat() for m in M])
        Q = np.mean(Q, axis=0)
        Q /= np.linalg.norm(Q)
        Q = transform.Rotation.from_quat(Q).as_matrix()

        # Reassemple into 4x4 transformation matrix
        tcp_zed = np.eye(4)
        tcp_zed[:3, :3] = Q
        tcp_zed[:3, 3] = tr

        return tcp_zed

    def __compute_calibration(
        self, measurements: List[TCPs], mode: str = "eye_to_hand"
    ):
        tcp_poses_in_base = [tcp.arm for tcp in measurements]
        board_poses_in_camera = [tcp.cam for tcp in measurements]

        calibration_errors: Dict[str, float] = {}
        calibration_poses: Dict[str, np.ndarray] = {}

        for name, method in cv2_CALIBRATION_METHODS.items():
            camera_pose, calibration_error = compute_calibration(
                board_poses_in_camera, tcp_poses_in_base, mode, method
            )
            calibration_poses[name] = camera_pose
            calibration_errors[name] = (
                float("inf") if calibration_error is None else calibration_error
            )

        methods = sorted(cv2_CALIBRATION_METHODS, key=lambda k: calibration_errors[k])
        return calibration_poses[methods[0]], calibration_errors[methods[0]]


def main():
    node = CameraCalibration()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
