import pickle
import sys
import time
from typing import Optional

import numpy as np
from pairo_butler.ur3_arms.ur3_solver import (
    UR3SophieSolver,
    UR3WilsonSolver,
)

import rospy as ros
from airo_butler.srv import PODService
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import (
    BooleanPOD,
    URStatePOD,
    make_pod_request,
    URPosePOD,
    UR3GripperPOD,
)
from pairo_butler.utils.tools import pyout

ARM_NAMES = ["wilson", "sophie"]
APPROACH_DISTANCE = 0.1
SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi


class UR3Client:
    def __init__(self, arm_name: str, name: str = "ur3_client"):
        assert (
            arm_name in ARM_NAMES
        ), f"Arm {arm_name} not available, choose from {ARM_NAMES}"
        self.arm_name = arm_name

        self.solver = (
            UR3SophieSolver() if self.arm_name == "sophie" else UR3WilsonSolver()
        )
        try:
            ros.wait_for_service("move_to_joint_configuration", timeout=5.0)
            ros.wait_for_service("move_to_tcp_pose", timeout=5.0)
            ros.wait_for_service("move_gripper", timeout=5.0)
            ros.wait_for_service("inverse_kinematics", timeout=5.0)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to UR5 server. Is it running?")
            ros.signal_shutdown("Cannot connect to UR5 server.")
            sys.exit(0)

        self.move_to_joint_configuration_service = ros.ServiceProxy(
            "move_to_joint_configuration", PODService
        )
        self.move_to_tcp_pose_service = ros.ServiceProxy("move_to_tcp_pose", PODService)
        self.move_gripper_service = ros.ServiceProxy("move_gripper", PODService)
        self.inverse_kinematics_service = ros.ServiceProxy(
            "inverse_kinematics", PODService
        )

        self.pose_sub: Optional[ros.Subscriber] = ros.Subscriber(
            f"/ur3_state_{self.arm_name}", PODMessage, self.__callback, queue_size=2
        )
        self.__joint_configuration: Optional[np.ndarray] = None
        self.__tcp_pose: Optional[np.ndarray] = None
        self.__gripper_width: Optional[float] = None

    def __callback(self, msg):
        msg: URStatePOD = pickle.loads(msg.data)

        self.__joint_configuration = msg.joint_configuration
        self.__tcp_pose = msg.tcp_pose
        self.__gripper_width = msg.gripper_width

    def move_to_joint_configuration(
        self,
        joint_configuration: np.ndarray,
        joint_speed: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        pod = URPosePOD(joint_configuration, self.arm_name, joint_speed, blocking)
        response = make_pod_request(
            self.move_to_joint_configuration_service, pod, BooleanPOD
        )
        return response.value

    def move_to_tcp_pose(
        self,
        tcp_pose: np.ndarray,
        joint_speed: Optional[float] = None,
        blocking: Optional[bool] = True,
    ) -> bool:
        pod = URPosePOD(tcp_pose, self.arm_name, joint_speed, blocking)
        response = make_pod_request(self.move_to_tcp_pose_service, pod, BooleanPOD)
        return response.value

    def move_gripper(self, width: float, blocking: bool = True) -> bool:
        pod = UR3GripperPOD(width, self.arm_name, blocking)
        response = make_pod_request(self.move_gripper_service, pod, BooleanPOD)
        return response.value

    def close_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("close", self.arm_name, blocking)
        response = make_pod_request(self.move_gripper_service, pod, BooleanPOD)
        return response.value

    def open_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("open", self.arm_name, blocking)
        response = make_pod_request(self.move_gripper_service, pod, BooleanPOD)
        return response.value

    def inverse_kinematics(self, tcp: np.ndarray, initial_config: Optional[np.ndarray]):
        pod = URStatePOD(
            tcp_pose=tcp,
            joint_configuration=initial_config,
            timestamp=ros.Time.now(),
            arm_name=self.arm_name,
        )
        response = make_pod_request(self.inverse_kinematics_service, pod, URStatePOD)
        return response.joint_configuration

    def get_tcp_pose(self, timeout=1):
        t0 = time.time()
        while self.__tcp_pose is None and time.time() - t0 < timeout:
            time.sleep(0.01)
        if self.__tcp_pose is None:
            raise TimeoutError
        return self.__tcp_pose

    def get_joint_configuration(self, timeout=1):
        t0 = time.time()
        while self.__joint_configuration is None and time.time() - t0 < timeout:
            time.sleep(0.01)
        if self.__joint_configuration is None:
            raise TimeoutError
        return self.__joint_configuration

    def get_gripper_width(self, timeout=1):
        t0 = time.time()
        while self.__gripper_width is None and time.time() - t0 < timeout:
            time.sleep(0.01)
        if self.__gripper_width is None:
            raise TimeoutError
        return self.__gripper_width

    def move_to_tcp_vertical_up(self, x: np.ndarray, speed: Optional[float] = None):
        tcp, initial_config = self.solver.solve_tcp_vertical_up(x)

        self.move_to_joint_configuration(initial_config, joint_speed=speed)

    def move_to_tcp_vertical_down(self, x: np.ndarray, speed: Optional[float] = None):
        tcp, initial_config = self.solver.solve_tcp_vertical_down(x)

        if tcp is None:
            self.move_to_joint_configuration(initial_config, joint_speed=speed)
        else:
            joint_config = self.inverse_kinematics(tcp, initial_config)
            if len(joint_config):
                self.move_to_joint_configuration(joint_config, joint_speed=speed)
            else:
                self.move_to_joint_configuration(initial_config, joint_speed=speed)

    def move_to_tcp_horizontal(
        self,
        x: np.ndarray,
        z: np.ndarray,
        speed: Optional[float] = None,
        flipped: bool = False,
        allow_flip: bool = True,
        blocking: bool = True,
    ):
        tcp, initial_config, flipped = self.solver.solve_tcp_horizontal(
            x, z, flipped=flipped, allow_flip=allow_flip
        )

        if tcp is None:
            self.move_to_joint_configuration(
                initial_config, joint_speed=speed, blocking=blocking
            )
        else:
            joint_config = self.inverse_kinematics(tcp, initial_config=initial_config)
            if len(joint_config):
                self.move_to_joint_configuration(
                    joint_config, joint_speed=speed, blocking=blocking
                )
            else:
                self.move_to_joint_configuration(
                    initial_config, joint_speed=speed, blocking=blocking
                )
        return flipped

    def grasp_horizontal(
        self,
        world_pos: np.ndarray,
        gripper_z_dir: np.ndarray,
        speed: Optional[float] = None,
    ):
        if self.arm_name == "sophie":
            joint_config_rest = SOPHIE_REST
        elif self.arm_name == "wilson":
            joint_config_rest = WILSON_REST
        else:
            raise ValueError(f'Arm name "{self.arm_name}" unknown.')
        self.move_to_joint_configuration(joint_config_rest)

        world_pos_approach: np.ndarray = world_pos - APPROACH_DISTANCE * gripper_z_dir
        _, approach_joint_config, flipped = self.solver.solve_tcp_horizontal(
            world_pos_approach, gripper_z_dir
        )

        if self.arm_name == "sophie":
            prepare_joint_config = np.copy(joint_config_rest)
            if np.rad2deg(approach_joint_config[0]) > 30:
                prepare_joint_config[0] = np.deg2rad(90)
            elif np.rad2deg(approach_joint_config[0] < -30):
                prepare_joint_config[0] = np.deg2rad(-90)

            self.open_gripper()
            self.move_to_joint_configuration(prepare_joint_config)
            self.move_to_joint_configuration(approach_joint_config)
            self.move_to_tcp_horizontal(
                world_pos, gripper_z_dir, flipped=flipped, allow_flip=False
            )
            self.close_gripper()

        else:
            raise NotImplementedError


def prt(A):
    angles = ", ".join([f"{np.rad2deg(angle):.0f}" for angle in A])
    return f"[{angles}]"
