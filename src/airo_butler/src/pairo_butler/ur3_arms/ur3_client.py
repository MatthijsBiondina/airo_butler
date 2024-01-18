import pickle
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
    UR3StatePOD,
    make_pod_request,
    UR3PosePOD,
    UR3GripperPOD,
)
from pairo_butler.utils.tools import pyout


class UR3Client:
    def __init__(self, left_or_right_arm: str, name: str = "ur3_client"):
        assert left_or_right_arm in ["left", "right"]
        self.side = left_or_right_arm

        self.solver = UR3SophieSolver() if self.side == "right" else UR3WilsonSolver()

        ros.wait_for_service("move_to_joint_configuration")
        self.move_to_joint_configuration_service = ros.ServiceProxy(
            "move_to_joint_configuration", PODService
        )

        ros.wait_for_service("move_to_tcp_pose")
        self.move_to_tcp_pose_service = ros.ServiceProxy("move_to_tcp_pose", PODService)

        ros.wait_for_service("move_gripper")
        self.move_gripper_service = ros.ServiceProxy("move_gripper", PODService)

        ros.wait_for_service("inverse_kinematics")
        self.inverse_kinematics_service = ros.ServiceProxy(
            "inverse_kinematics", PODService
        )

        self.pose_sub: Optional[ros.Subscriber] = ros.Subscriber(
            f"/ur3_state_{self.side}", PODMessage, self.__callback, queue_size=2
        )
        self.__joint_configuration: Optional[np.ndarray] = None
        self.__tcp_pose: Optional[np.ndarray] = None
        self.__gripper_width: Optional[float] = None

    def __callback(self, msg):
        msg: UR3StatePOD = pickle.loads(msg.data)

        self.__joint_configuration = msg.joint_configuration
        self.__tcp_pose = msg.tcp_pose
        self.__gripper_width = msg.gripper_width

    def move_to_joint_configuration(
        self,
        joint_configuration: np.ndarray,
        joint_speed: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        pod = UR3PosePOD(joint_configuration, self.side, joint_speed, blocking)
        response = make_pod_request(
            self.move_to_joint_configuration_service, pod, BooleanPOD
        )
        return response.value

    def move_to_tcp_pose(self, tcp_pose: np.ndarray, breakpoint) -> bool:
        pod = UR3PosePOD(tcp_pose, self.side, joint_speed, blocking)
        response = make_pod_request(self.move_to_tcp_pose_service, pod, BooleanPOD)
        return response.value

    def move_gripper(self, width: float, blocking: bool = True) -> bool:
        pod = UR3GripperPOD(width, self.side, blocking)
        response = make_pod_request(self.move_gripper, pod, BooleanPOD)
        return response.value

    def close_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("close", self.side, blocking)
        response = make_pod_request(self.move_gripper_service, pod, BooleanPOD)
        return response.value

    def open_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("open", self.side, blocking)
        response = make_pod_request(self.move_gripper_service, pod, BooleanPOD)
        return response.value

    def inverse_kinematics(self, tcp: np.ndarray, initial_config: Optional[np.ndarray]):
        pod = UR3StatePOD(
            tcp_pose=tcp,
            joint_configuration=initial_config,
            timestamp=ros.Time.now(),
            side=self.side,
        )
        response = make_pod_request(self.inverse_kinematics_service, pod, UR3StatePOD)
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

    def grasp_down(self, x: np.ndarray, speed: Optional[float] = None):
        SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
        WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi
        pyout(f"Sophie rest: {prt(SOPHIE_REST)}")
        pyout(f"Wilson rest: {prt(WILSON_REST)}")
        tcp, initial_config = self.solver.solve_tcp_vertical_down(x)

        ros.loginfo(initial_config)
        pyout(f"Vertical: {prt(initial_config)}")

        joint_config = self.inverse_kinematics(tcp, initial_config)
        if len(joint_config):
            self.move_to_joint_configuration(joint_config, joint_speed=speed)
        else:
            self.move_to_joint_configuration(initial_config, joint_speed=speed)

    def grasp_horizontal(
        self, x: np.ndarray, z: np.ndarray, speed: Optional[float] = None
    ):
        tcp, initial_config = self.solver.solve_tcp_horizontal(x, z)

        SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
        WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi

        pyout(f"Target: {prt(initial_config)}")
        pyout(f"Sophie rest: {prt(SOPHIE_REST)}")
        pyout(f"Wilson rest: {prt(WILSON_REST)}")

        self.move_to_joint_configuration(initial_config, joint_speed=0.1)

        ros.loginfo(initial_config)
        pyout()


def prt(A):
    angles = ", ".join([f"{np.rad2deg(angle):.0f}" for angle in A])
    return f"[{angles}]"
