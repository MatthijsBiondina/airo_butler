import os
import pickle
import threading
from typing import Dict, Optional

from munch import Munch
import numpy as np
from pairo_butler.utils.tools import load_config, pyout
from pairo_butler.utils.pods import (
    BooleanPOD,
    DualTrajectoryPOD,
    SingleTrajectoryPOD,
    URGripperPOD,
    URPosePOD,
    URStatePOD,
    publish_pod,
)
import rospy as ros
from airo_robots.manipulators import URrtde
from airo_robots.grippers import Robotiq2F85
from airo_butler.msg import PODMessage
from airo_butler.srv import PODService, PODServiceResponse

np.set_printoptions(precision=2, suppress=True)

ARM_Y_DEFAULT = 0.45


class UR5e_server:
    PUBLISH_RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "ur5e_server"):
        self.config: Munch = load_config()

        self.sophie = URrtde(self.config.ip_sophie, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(self.config.ip_sophie)
        self.wilson = URrtde(self.config.ip_wilson, URrtde.UR3E_CONFIG)
        self.wilson.gripper = Robotiq2F85(self.config.ip_wilson)

        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.sophie_publisher_name: str = "/ur5e_sophie"
        self.wilson_publisher_name: str = "/ur5e_wilson"

        # Placeholders for publishers
        self.sophie_publisher: Optional[ros.Publisher] = None
        self.wilson_publisher: Optional[ros.Publisher] = None

        # Initialize services
        self.services: Dict[str, ros.Service] = self.__initialize_services()

        self.transform_sophie_to_world = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, -ARM_Y_DEFAULT],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.transform_wilson_to_world = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, ARM_Y_DEFAULT],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.sophie_publisher = ros.Publisher(
            self.sophie_publisher_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.wilson_publisher = ros.Publisher(
            self.wilson_publisher_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):

        while not ros.is_shutdown():
            timestamp = ros.Time.now()

            tcp_sophie_in_sophie_frame = self.sophie.get_tcp_pose()
            tcp_sophie_in_world_frame = (
                self.transform_sophie_to_world @ tcp_sophie_in_sophie_frame
            )
            tcp_wilson_in_wilson_frame = self.wilson.get_tcp_pose()
            tcp_wilson_in_world_frame = (
                self.transform_wilson_to_world @ tcp_wilson_in_wilson_frame
            )
            sophie_gripper_width = (
                0.08
                if self.sophie.gripper is None
                else self.sophie.gripper.get_current_width()
            )
            pod_sophie = URStatePOD(
                tcp_pose=tcp_sophie_in_world_frame,
                joint_configuration=self.sophie.get_joint_configuration(),
                gripper_width=sophie_gripper_width,
                timestamp=timestamp,
                arm_name="sophie",
            )
            wilson_gripper_width = (
                0.08
                if self.wilson.gripper is None
                else self.wilson.gripper.get_current_width()
            )
            pod_wilson = URStatePOD(
                tcp_pose=tcp_wilson_in_world_frame,
                joint_configuration=self.wilson.get_joint_configuration(),
                gripper_width=wilson_gripper_width,
                timestamp=timestamp,
                arm_name="wilson",
            )

            publish_pod(self.sophie_publisher, pod_sophie)
            publish_pod(self.wilson_publisher, pod_wilson)

            self.rate.sleep()

    # PRIVATE METHODS
    def __initialize_services(self) -> Dict[str, ros.Service]:
        services = {
            "move_to_joint_configuration": ros.Service(
                "move_to_joint_configuration",
                PODService,
                self.__move_to_joint_configuration,
            ),
            "execute_trajectory": ros.Service(
                "execute_trajectory",
                PODService,
                self.__execute_trajectory,
            ),
            "move_gripper": ros.Service(
                "move_gripper",
                PODService,
                self.__move_gripper,
            ),
            "interrupt": ros.Service(
                "interrupt",
                PODService,
                self.__interrupt,
            ),
        }

        return services

    def __move_to_joint_configuration(self, req):
        try:
            pod: URPosePOD = pickle.loads(req.pod)

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

    def __execute_trajectory(self, req):
        try:
            pod: DualTrajectoryPOD = pickle.loads(req.pod)

            assert np.all(
                np.isclose(
                    pod.path_wilson[0],
                    self.wilson.get_joint_configuration(),
                    atol=1e-1,
                )
            )

            assert np.all(
                np.isclose(
                    pod.path_sophie[0],
                    self.sophie.get_joint_configuration(),
                    atol=1e-1,
                )
            )

            for joints_wilson, joints_sophie in zip(pod.path_wilson, pod.path_sophie):
                wilson_servo = self.wilson.servo_to_joint_configuration(
                    joints_wilson, pod.period
                )
                sophie_servo = self.sophie.servo_to_joint_configuration(
                    joints_sophie, pod.period
                )
                wilson_servo.wait()
                sophie_servo.wait()

            return_value = True
        except Exception as e:
            raise e
            ros.logwarn(f"An exception occurred: {e}")
            return_value = False

        response = PODServiceResponse()
        response.pod = pickle.dumps(BooleanPOD(return_value))
        return response

    def __move_gripper(self, req):
        try:
            pod: URGripperPOD = pickle.loads(req.pod)

            # Determine the arm based on the side
            assert pod.arm_name in ["wilson", "sophie"]
            arm = self.wilson if pod.arm_name == "wilson" else self.sophie

            if arm.gripper is None:
                response = PODServiceResponse()
                response.pod = pickle.dumps(BooleanPOD(False))
                return response

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

    def __interrupt(self, req):
        raise NotImplementedError


def main():
    node = UR5e_server()
    node.start_ros()
    threading.Thread(target=node.run, daemon=True).start()
    ros.spin()


if __name__ == "__main__":
    main()
