import pickle
import time
from typing import Optional

import numpy as np

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

        ros.wait_for_service("move_to_joint_configuration")
        self.move_to_joint_configuration_service = ros.ServiceProxy(
            "move_to_joint_configuration", PODService
        )

        ros.wait_for_service("move_to_tcp_pose")
        self.move_to_tcp_pose_service = ros.ServiceProxy("move_to_tcp_pose", PODService)

        ros.wait_for_service("move_gripper")
        self.move_gripper_service = ros.ServiceProxy("move_gripper", PODService)

        self.pose_sub: Optional[ros.Subscriber] = ros.Subscriber(
            f"/ur3_state_{self.side}", PODMessage, self.__callback, queue_size=2
        )
        self.__joint_configuration: Optional[np.ndarray] = None
        self.__tcp_pose: Optional[np.ndarray] = None

    def __callback(self, msg):
        msg: UR3StatePOD = pickle.loads(msg.data)

        self.__joint_configuration = msg.joint_configuration
        self.__tcp_pose = msg.tcp_pose

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


# if __name__ == "__main__":
#     ros.init_node(f"UR3_arms_client", log_level=ros.INFO)

#     left_arm = UR3Client("left")
#     left_arm.move_to_joint_configuration(POSE_LEFT_PRESENT)

#     right_arm = UR3Client("right")
#     right_arm.move_to_joint_configuration(POSE_RIGHT_SCAN1)
#     time.sleep(10)
