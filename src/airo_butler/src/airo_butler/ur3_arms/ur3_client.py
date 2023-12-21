import pickle
from typing import Optional

import numpy as np

import rospy as ros
from airo_butler.srv import PODService
from airo_butler.utils.pods import BooleanPOD, \
    make_pod_request, UR3PosePOD, UR3GripperPOD
from airo_butler.utils.tools import pyout
from airo_butler.ur3_arms.ur3_constants import POSE_LEFT_REST, \
    POSE_LEFT_PRESENT, POSE_RIGHT_SCAN1


class UR3Client:
    def __init__(self, left_or_right_arm: str,
                 name: str = 'ur3_client'):
        assert left_or_right_arm in ['left', 'right']
        self.side = left_or_right_arm

        ros.init_node(f"{name}_{self.side}", log_level=ros.INFO,
                      anonymous=True)

        ros.wait_for_service("move_to_joint_configuration")
        self.move_to_joint_configuration_service = ros.ServiceProxy(
            "move_to_joint_configuration", PODService)

        ros.wait_for_service("move_to_tcp_pose")
        self.move_to_tcp_pose_service = ros.ServiceProxy(
            "move_to_tcp_pose", PODService)

        ros.wait_for_service("move_gripper")
        self.move_gripper_service = ros.ServiceProxy(
            "move_gripper", PODService)

    def move_to_joint_configuration(self,
                                    joint_configuration: np.ndarray,
                                    joint_speed: Optional[float] = None,
                                    blocking: bool = True
                                    ) -> bool:
        pod = UR3PosePOD(
            joint_configuration, self.side, joint_speed, blocking)
        response = make_pod_request(self.move_to_joint_configuration_service,
                                    pod, BooleanPOD)
        return response.value

    def move_to_tcp_pose(self,
                         tcp_pose: np.ndarray,
                         joint_speed: Optional[float] = None,
                         blocking: bool = True
                         ) -> bool:
        pod = UR3PosePOD(
            tcp_pose, self.side, joint_speed, blocking)
        response = make_pod_request(self.move_to_tcp_pose_service,
                                    pod, BooleanPOD)
        return response.value

    def move_gripper(self, width: float, blocking: bool = True) -> bool:
        pod = UR3GripperPOD(width, self.side, blocking)
        response = make_pod_request(self.move_gripper, pod, BooleanPOD)
        return response.value

    def close_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("close", self.side, blocking)
        response = make_pod_request(self.move_gripper_service, pod,
                                    BooleanPOD)
        return response.value

    def open_gripper(self, blocking: bool = True) -> bool:
        pod = UR3GripperPOD("open", self.side, blocking)
        response = make_pod_request(self.move_gripper_service, pod,
                                    BooleanPOD)
        return response.value


if __name__ == "__main__":
    left_arm = UR3Client("left")
    left_arm.move_to_joint_configuration(POSE_LEFT_PRESENT)


    right_arm = UR3Client("right")
    right_arm.move_to_joint_configuration(POSE_RIGHT_SCAN1)
    pyout()
