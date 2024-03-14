import pickle
import sys

import numpy as np
from pairo_butler.utils.pods import (
    BooleanPOD,
    DualTrajectoryPOD,
    SingleTrajectoryPOD,
    URGripperPOD,
    URPosePOD,
    URStatePOD,
    make_pod_request,
)
import rospy as ros
from typing import Any, Callable, Optional
from pairo_butler.utils.tools import load_config, pyout
from airo_butler.srv import PODService
from airo_butler.msg import PODMessage


class UR5eClient:
    QUEUE_SIZE = 2
    PATIENCE = 5.0

    def __init__(self, arm_name: str, name: str = "ur5_client"):
        self.config = load_config()

        assert arm_name in self.config.arm_names, f"Unknown arm: {arm_name}"
        self.arm_name = arm_name

        # Wait for services
        self.__service_move_to_joint_configuration: Optional[Callable[[Any], Any]]
        self.__service_execute_trajectory: Optional[Callable[[Any], Any]]
        self.__service_move_gripper: Optional[Callable[[Any], Any]]
        self.__service_interrupt: Optional[Callable[[Any], Any]]
        self.__wait_for_services()

        self.__state_subscriber: Optional[ros.Subscriber] = None
        self.__state_joint_configuration: Optional[np.ndarray] = None
        self.__state_tcp_pose: Optional[np.ndarray] = None
        self.__state_gripper_width: Optional[float] = None
        self.__initialize_subscribers()

    def get_joint_configuration(self):
        t_start = ros.Time.now()
        while (
            self.__state_joint_configuration is None
            and ros.Time.now() - t_start < ros.Duration(secs=self.PATIENCE)
        ):
            ros.sleep(duration=ros.Duration(nsecs=1000))
        if self.__state_joint_configuration is None:
            raise TimeoutError
        return self.__state_joint_configuration

    def get_tcp_pose(self):
        t_start = ros.Time.now()
        while self.__state_tcp_pose is None and ros.Time.now() - t_start < ros.Duration(
            secs=self.PATIENCE
        ):
            ros.sleep(duration=ros.Duration(nsecs=1000))
        if self.__state_tcp_pose is None:
            raise TimeoutError
        return self.__state_tcp_pose

    def get_gripper_width(self):
        t_start = ros.Time.now()
        while (
            self.__state_gripper_width is None
            and ros.Time.now() - t_start < ros.Duration(secs=self.PATIENCE)
        ):
            ros.sleep(duration=ros.Duration(nsecs=1000))
        if self.__state_gripper_width is None:
            raise TimeoutError
        return self.__state_gripper_width

    def move_to_joint_configuration(
        self,
        joint_configuration: np.ndarray,
        joint_speed: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        pod = URPosePOD(joint_configuration, self.arm_name, joint_speed, blocking)
        response = make_pod_request(
            self.__service_move_to_joint_configuration, pod, BooleanPOD
        )
        return response.value

    def execute_plan(self, plan: DualTrajectoryPOD):
        response = make_pod_request(self.__service_execute_trajectory, plan, BooleanPOD)
        return response.value

    def close_gripper(self, blocking: bool = True) -> bool:
        pod = URGripperPOD("close", self.arm_name, blocking)
        response = make_pod_request(self.__service_move_gripper, pod, BooleanPOD)
        return response.value

    def open_gripper(self, blocking: bool = True) -> bool:
        pod = URGripperPOD("open", self.arm_name, blocking)
        response = make_pod_request(self.__service_move_gripper, pod, BooleanPOD)
        return response.value

    # SERVICE CLIENTS

    def __wait_for_services(self):
        try:
            ros.wait_for_service("move_to_joint_configuration", timeout=self.PATIENCE)
            ros.wait_for_service("execute_trajectory", timeout=self.PATIENCE)
            ros.wait_for_service("move_gripper", timeout=self.PATIENCE)
            ros.wait_for_service("interrupt", timeout=self.PATIENCE)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to UR5 server. Is it running?")
            ros.signal_shutdown("Cannot connect to UR5 server.")
            sys.exit(0)

        self.__service_move_to_joint_configuration = ros.ServiceProxy(
            "move_to_joint_configuration", PODService
        )
        self.__service_execute_trajectory = ros.ServiceProxy(
            "execute_trajectory", PODService
        )
        self.__service_move_gripper = ros.ServiceProxy("move_gripper", PODService)
        self.__service_interrupt = ros.ServiceProxy("interrupt", PODService)

    # SUBSCRIBERS

    def __initialize_subscribers(self):
        self.__state_subscriber = ros.Subscriber(
            f"/ur5e_{self.arm_name}",
            PODMessage,
            self.__subscriber_callback,
            queue_size=self.QUEUE_SIZE,
        )

    def __subscriber_callback(self, msg):
        msg: URStatePOD = pickle.loads(msg.data)

        self.__state_joint_configuration = msg.joint_configuration
        self.__state_tcp_pose = msg.tcp_pose
        self.__state_gripper_width = msg.gripper_width
