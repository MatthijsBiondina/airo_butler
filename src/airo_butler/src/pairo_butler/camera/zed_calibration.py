from copy import deepcopy
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2

import numpy as np
import rospkg
from pairo_butler.ur3_arms.ur3_constants import WILSON_SLEEP, SOPHIE_SLEEP
from pairo_butler.utils.tools import pyout
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
SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi


# Wilson poses
WILSON_CALIBRATION_POSES = (
    np.array(
        [
            [+0.60, -0.00, -0.40, -0.60, +1.05, -0.95],
            [+0.60, -0.00, -0.40, -0.60, +1.05, -1.10],
            [+0.60, -0.00, -0.40, -0.60, +0.95, -0.95],
            [+0.60, -0.00, -0.40, -0.60, +0.95, -1.10],
            [+0.60, -0.00, -0.25, -0.25, +0.55, -0.55],
            [+0.60, -0.00, -0.25, -0.25, +0.55, -0.65],
            [+0.60, -0.00, -0.25, -0.25, +0.45, -0.55],
            [+0.60, -0.00, -0.25, -0.25, +0.45, -0.65],
            [+0.50, -0.00, -0.40, -0.60, +0.05, +0.05],
            [+0.50, -0.00, -0.40, -0.60, +0.05, -0.05],
            [+0.50, -0.00, -0.40, -0.60, -0.05, +0.05],
            [+0.50, -0.00, -0.40, -0.60, -0.05, -0.05],
            [+0.60, -0.40, +0.25, -0.45, -0.45, -0.35],
            [+0.60, -0.40, +0.25, -0.45, -0.45, -0.45],
            [+0.60, -0.40, +0.25, -0.45, -0.55, -0.35],
            [+0.60, -0.40, +0.25, -0.45, -0.55, -0.45],
        ]
    )
    * np.pi
)
WILSON_PRESENT_TO_SOPHIE = np.array([+0.50, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi
WILSON_TRANSFER_POSE = np.array([+0.50, -0.20, -0.00, -0.80, +0.10, +0.00]) * np.pi

SOPHIE_EYE_IN_HAND_POSES = (
    np.array(
        [
            [-0.50, -1.00, +0.25, -0.25, -0.30, +0.00],
            [-0.33, -1.00, +0.30, -0.30, -0.35, +0.00],
            [-0.16, -1.00, +0.40, -0.40, -0.42, +0.00],
            [+0.00, -1.00, +0.50, -0.50, -0.50, +0.00],
            [+0.00, -0.95, +0.45, -0.45, -0.50, +0.00],
            [+0.00, -0.90, +0.40, -0.42, -0.50, +0.00],
            [+0.00, -0.80, +0.30, -0.40, -0.50, +0.00],
            [+0.00, -0.75, +0.15, -0.30, -0.50, +0.00],
            [+0.00, -0.60, +0.00, -0.25, -0.50, +0.00],
            [+0.60, -1.00, +0.25, -0.25, -0.85, +0.00],
            [+0.45, -1.00, +0.30, -0.30, -0.80, +0.00],
            [+0.30, -1.00, +0.35, -0.35, -0.70, +0.00],
            # [+0.15, -1.00, +0.43, -0.43, -0.62, +0.00],
        ]
    )
    * np.pi
)
SOPHIE_LOOK_BEFORE_TRANSFER = (
    np.array([-0.40, -0.87, +0.00, -0.13, -0.25, +0.00]) * np.pi
)

# todo: debug
WILSON_REST = WILSON_TRANSFER_POSE
SOPHIE_REST = SOPHIE_LOOK_BEFORE_TRANSFER

# States
STATE_STARTUP = 0
STATE_GRAB_CHARUCO_BOARD = 1
STATE_CALIBRATE_ZED_WILSON = 2
STATE_CALIBRATE_RS2_SOPHIE = 3
STATE_TRANSFER_CHARUCO_BOARD = 4


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
        self.zed_sub: Optional[ros.Subscriber] = None
        self.rs2_sub: Optional[ros.Subscriber] = None
        self.wilson: Optional[UR3Client] = None
        self.sophie: Optional[UR3Client] = None

        # Placeholders:
        self.zed_pod: Optional[ZEDPOD] = None
        self.rs2_pod: Optional[ImagePOD] = None
        self.state: str = STATE_STARTUP

        # Filesystem
        self.data_root = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.zed_sub = ros.Subscriber(
            "/zed2i", PODMessage, self.__zed_sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.rs2_sub = ros.Subscriber(
            "/color_frame",
            PODMessage,
            self.__rs2_sub_callback,
            queue_size=self.QUEUE_SIZE,
        )
        self.wilson = UR3Client("left")
        self.sophie = UR3Client("right")
        ros.loginfo(f"{self.node_name}: OK!")

    def __zed_sub_callback(self, msg: PODMessage):
        self.zed_pod = pickle.loads(msg.data)

    def __rs2_sub_callback(self, msg: PODMessage):
        self.rs2_pod = pickle.loads(msg.data)

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
            else:
                ros.logwarn(f"Unknown state: {self.state}")
            self.rate.sleep()

    def __startup(self):
        self.sophie.move_to_joint_configuration(SOPHIE_REST)
        self.wilson.move_to_joint_configuration(WILSON_REST)
        self.wilson.close_gripper()
        time.sleep(2)
        return STATE_GRAB_CHARUCO_BOARD

    def __grab(self):
        while not ros.is_shutdown() and self.wilson.get_gripper_width() < 0.002:
            self.wilson.open_gripper()
            time.sleep(2)
            self.wilson.close_gripper()
            time.sleep(2)
        # return STATE_CALIBRATE_RS2_SOPHIE
        return STATE_TRANSFER_CHARUCO_BOARD

    def __calibrate_wilson(self):
        measurements: List[TCPs] = []

        for joint_config in WILSON_CALIBRATION_POSES:
            self.wilson.move_to_joint_configuration(joint_config)
            measurements.append(
                TCPs(
                    tcp_cam=self.__wait_for_measurements(ros.Time.now()),
                    tcp_arm=self.wilson.get_tcp_pose(),
                )
            )

        zed_pose, error = self.__compute_calibration(measurements, mode="eye_to_hand")
        ros.loginfo(f"Calibrated Wilson with error: {error:.3f}")
        save_path = self.data_root / "T_zed_wilson.npy"
        np.save(save_path, zed_pose)

        self.wilson.move_to_joint_configuration(WILSON_REST)

        return STATE_CALIBRATE_RS2_SOPHIE

    def __calibrate_sophie_rs2(self):
        self.wilson.move_to_joint_configuration(WILSON_PRESENT_TO_SOPHIE)

        measurements: List[TCPs] = []
        for joint_config in SOPHIE_EYE_IN_HAND_POSES:
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

        self.wilson.move_to_joint_configuration(WILSON_REST)
        self.sophie.move_to_joint_configuration(SOPHIE_REST)

        return STATE_TRANSFER_CHARUCO_BOARD

    def __transfer_charuco_board(self):
        self.wilson.move_to_joint_configuration(WILSON_TRANSFER_POSE)
        self.sophie.move_to_joint_configuration(SOPHIE_LOOK_BEFORE_TRANSFER)

        # location of board relative to wilson
        T_charuco_zed = self.__wait_for_measurements(ros.Time.now(), camera="zed")
        T_zed_wilson = np.load(self.data_root / "T_zed_wilson.npy")
        T_charuco_wilson = T_zed_wilson @ T_charuco_zed
        pyout(f"T_charuco_wilson:\n{T_charuco_wilson}")
        pyout(f"Wilson TCP:\n{self.wilson.get_tcp_pose()}")

        # location of board relative to sophie
        T_charuco_rs2 = self.__wait_for_measurements(ros.Time.now(), camera="rs2")
        T_rs2_tcp_sophie = np.load(self.data_root / "T_rs2_tcp_sophie.npy")
        T_tcp_sophie = self.sophie.get_tcp_pose()
        T_charuco_sophie = T_tcp_sophie @ T_rs2_tcp_sophie @ T_charuco_rs2
        pyout(f"T_charuco_sophie:\n{T_charuco_sophie}")
        pyout(f"Sophie TCP:\n{T_tcp_sophie}")

        ros.signal_shutdown("debug")
        sys.exit(0)

        pyout()

    def __wait_for_measurements(self, t0: ros.Time, timeout=10, camera: str = "zed"):
        t_start = time.time()
        timestamps: List[ros.Time] = []
        imgs: List[np.ndarray] = []
        TCP: List[np.ndarray] = []

        while not ros.is_shutdown() and len(imgs) < self.CHARUCO_FRAMES:
            if time.time() - t_start > timeout:
                return None
            if camera == "zed":
                pod = deepcopy(self.zed_pod)
                img = (pod.rgb_image * 255)[..., ::-1].astype(np.uint8)
            elif camera == "rs2":
                pod = deepcopy(self.rs2_pod)
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
