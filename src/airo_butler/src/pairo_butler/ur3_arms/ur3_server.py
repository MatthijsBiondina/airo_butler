#!/usr/bin/env python3
import os
import pickle
from typing import Dict, Optional
import sys

print(f"Python version: {sys.version}")

import numpy as np
from airo_robots.grippers import Robotiq2F85
from airo_robots.manipulators import URrtde
from pairo_butler.utils.tools import pyout

import rospy as ros

from airo_butler.msg import PODMessage
from airo_butler.srv import PODService, PODServiceResponse
from pairo_butler.ur3_arms.ur3_constants import IP_RIGHT_UR3, IP_LEFT_UR3
from pairo_butler.ur3_arms.ur3_utils import convert_homegeneous_pose_to_rotvec_pose
from pairo_butler.utils.pods import POD, UR3StatePOD, publish_pod
from pairo_butler.utils.pods import BooleanPOD, UR3PosePOD, UR3GripperPOD

RARM_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
LARM_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi


class UR3_server:
    PUBLISH_RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, ip_right: str, ip_left: str, name: str = "ur3_server"):
        self.arm_right = URrtde(ip_right, URrtde.UR3E_CONFIG)
        self.arm_right.gripper = Robotiq2F85(ip_right)
        self.arm_left = URrtde(ip_left, URrtde.UR3E_CONFIG)
        self.arm_left.gripper = Robotiq2F85(ip_left)

        self.node_name: str = f"{name}"
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.left_pub_name: str = "/ur3_state_left"
        self.right_pub_name: str = "/ur3_state_right"

        # Placeholder for publisher
        self.pub_left: Optional[ros.Publisher] = None
        self.pub_right: Optional[ros.Publisher] = None

        self.services: Dict[str, ros.Service] = self.__init_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.pub_left = ros.Publisher(
            self.left_pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.pub_right = ros.Publisher(
            self.right_pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo("UR3_server: OK!")

    def run(self):
        while not ros.is_shutdown():
            timestamp = ros.Time.now()
            pod_left = UR3StatePOD(
                self.arm_left.get_tcp_pose(),
                self.arm_left.get_joint_configuration(),
                self.arm_left.gripper.get_current_width(),
                timestamp,
            )
            pod_right = UR3StatePOD(
                self.arm_right.get_tcp_pose(),
                self.arm_right.get_joint_configuration(),
                self.arm_right.gripper.get_current_width(),
                timestamp,
            )

            publish_pod(self.pub_left, pod_left)
            publish_pod(self.pub_right, pod_right)

            self.rate.sleep()

    def __init_services(self) -> Dict[str, ros.Service]:
        services = {
            "move_to_joint_configuration": ros.Service(
                "move_to_joint_configuration",
                PODService,
                self.move_to_joint_configuration,
            ),
            "move_to_tcp_pose": ros.Service(
                "move_to_tcp_pose", PODService, self.move_to_tcp_pose
            ),
            "move_gripper": ros.Service("move_gripper", PODService, self.move_gripper),
            "inverse_kinematics": ros.Service(
                "inverse_kinematics", PODService, self.inverse_kinematics
            ),
        }

        return services

    def inverse_kinematics(self, req):
        if True:
            pod: UR3StatePOD = pickle.loads(req.pod)

            arm = self.arm_right if pod.side == "right" else self.arm_left
            tcp_rotvec_pose = convert_homegeneous_pose_to_rotvec_pose(pod.tcp_pose)
            q_near = pod.joint_configuration
            joint_config = arm.rtde_control.getInverseKinematics(
                tcp_rotvec_pose, q_near
            )

        response = PODServiceResponse()
        response.pod = pickle.dumps(
            UR3StatePOD(
                tcp_pose=pod.tcp_pose,
                joint_configuration=joint_config,
                side=pod.side,
                timestamp=ros.Time.now(),
            )
        )
        return response

    def move_to_joint_configuration(self, req):
        try:
            pod: UR3PosePOD = pickle.loads(req.pod)

            if pod.side == "left":
                action = self.arm_left.move_to_joint_configuration(
                    pod.pose, pod.joint_speed
                )
            elif pod.side == "right":
                action = self.arm_right.move_to_joint_configuration(
                    pod.pose, pod.joint_speed
                )
            else:
                raise ValueError(f"Invalid side: {pod.side}")

            if pod.blocking:
                action.wait()

            return_value = True
        except Exception as e:
            return_value = False

        response = PODServiceResponse()
        response.pod = pickle.dumps(BooleanPOD(return_value))
        return response

    def move_to_tcp_pose(self, req):
        try:
            pod: UR3PosePOD = pickle.loads(req.pod)

            if pod.side == "left":
                action = self.arm_left.move_to_tcp_pose(pod.pose, pod.joint_speed)
            elif pod.side == "right":
                action = self.arm_right.move_to_tcp_pose(pod.pose, pod.joint_speed)
            else:
                raise ValueError(f"Invalid side: {pod.side}")

            if pod.blocking:
                action.wait()
            return_value = True
        except Exception as e:
            return_value = False

        response = PODServiceResponse()
        response.pod = pickle.dumps(BooleanPOD(return_value))
        return response

    def move_gripper(self, req):
        try:
            pod: UR3GripperPOD = pickle.loads(req.pod)

            # Determine the arm based on the side
            assert pod.side in ["left", "right"]
            arm = self.arm_left if pod.side == "left" else self.arm_right

            # Determine the width to move the gripper to
            if isinstance(pod.pose, float):
                width = np.clip(pod.pose, 0.0, arm.gripper.gripper_specs.max_width)
            else:
                assert pod.pose in ["open", "close"]
                width = (
                    0.0 if pod.pose == "close" else arm.gripper.gripper_specs.max_width
                )

            arm.gripper.move(width)
            return_value = True
        except Exception as e:
            return_value = False
        response = PODServiceResponse()
        response.pod = pickle.dumps(BooleanPOD(return_value))
        return response


def main():
    server = UR3_server(IP_RIGHT_UR3, IP_LEFT_UR3)
    server.start_ros()
    server.run()


if __name__ == "__main__":
    main()
