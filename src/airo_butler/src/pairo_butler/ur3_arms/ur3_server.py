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
        self.sophie = URrtde(ip_right, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(ip_right)
        self.wilson = URrtde(ip_left, URrtde.UR3E_CONFIG)
        self.wilson.gripper = Robotiq2F85(ip_left)

        self.node_name: str = f"{name}"
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.wilson_pub_name: str = "/ur3_state_wilson"
        self.sophie_pub_name: str = "/ur3_state_sophie"

        # Placeholder for publisher
        self.pub_wilson: Optional[ros.Publisher] = None
        self.pub_sophie: Optional[ros.Publisher] = None

        self.services: Dict[str, ros.Service] = self.__init_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.pub_wilson = ros.Publisher(
            self.wilson_pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.pub_sophie = ros.Publisher(
            self.sophie_pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo("UR3_server: OK!")

    def run(self):
        while not ros.is_shutdown():
            timestamp = ros.Time.now()
            pod_left = UR3StatePOD(
                tcp_pose=self.wilson.get_tcp_pose(),
                joint_configuration=self.wilson.get_joint_configuration(),
                gripper_width=self.wilson.gripper.get_current_width(),
                timestamp=timestamp,
                arm_name="wilson",
            )
            pod_right = UR3StatePOD(
                tcp_pose=self.sophie.get_tcp_pose(),
                joint_configuration=self.sophie.get_joint_configuration(),
                gripper_width=self.sophie.gripper.get_current_width(),
                timestamp=timestamp,
                arm_name="sophie",
            )

            publish_pod(self.pub_wilson, pod_left)
            publish_pod(self.pub_sophie, pod_right)

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

            assert pod.arm_name in [
                "sophie",
                "wilson",
            ], f"Invalid arm name: {pod.arm_name}"

            arm = self.sophie if pod.arm_name == "sophie" else self.wilson
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
                arm_name=pod.arm_name,
                timestamp=ros.Time.now(),
            )
        )
        return response

    def move_to_joint_configuration(self, req):
        try:
            pod: UR3PosePOD = pickle.loads(req.pod)

            if pod.arm_name == "wilson":
                action = self.wilson.move_to_joint_configuration(
                    pod.pose, pod.joint_speed
                )
            elif pod.arm_name == "sophie":
                action = self.sophie.move_to_joint_configuration(
                    pod.pose, pod.joint_speed
                )
            else:
                raise ValueError(f"Invalid arm: {pod.arm_name}")

            if pod.blocking:
                action.wait(timeout=300.0)

            return_value = True
        except Exception as e:
            return_value = False

        response = PODServiceResponse()
        response.pod = pickle.dumps(BooleanPOD(return_value))
        return response

    def move_to_tcp_pose(self, req):
        try:
            pod: UR3PosePOD = pickle.loads(req.pod)

            if pod.side == "wilson":
                action = self.wilson.move_to_tcp_pose(pod.pose, pod.joint_speed)
            elif pod.side == "sophie":
                action = self.sophie.move_to_tcp_pose(pod.pose, pod.joint_speed)
            else:
                raise ValueError(f"Invalid side: {pod.side}")

            if pod.blocking:
                action.wait(timeout=300.0)
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
            assert pod.arm_name in ["wilson", "sophie"]
            arm = self.wilson if pod.arm_name == "wilson" else self.sophie

            # Determine the width to move the gripper to
            if isinstance(pod.pose, float):
                width = np.clip(pod.pose, 0.0, arm.gripper.gripper_specs.max_width)
            else:
                assert pod.pose in ["open", "close"]
                width = (
                    0.0 if pod.pose == "close" else arm.gripper.gripper_specs.max_width
                )

            arm.gripper.move(width, speed=0.15, force=250).wait(timeout=300.0)
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
