import sys

import numpy as np
from pairo_butler.utils.pods import (
    DualJointsPOD,
    DualTrajectoryPOD,
    DualTCPPOD,
    make_pod_request,
)
import rospy as ros
from typing import Any, Callable, Optional, Tuple
from pairo_butler.utils.tools import load_config, pyout
from airo_butler.srv import PODService


class OMPLClient:
    QUEUE_SIZE = 2
    PATIENCE = 5.0

    def __init__(self, name: str = "ompl_client"):
        self.name = name
        self.config = load_config()

        self.__service_plan_to_tcp_pose: Callable[[Any], Any]
        # self.__service_plan_grasp_trajectory: Callable[[Any], Any]
        self.__wait_for_services()

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)

    def __wait_for_services(self):
        try:
            ros.wait_for_service("plan_to_tcp_pose", timeout=self.PATIENCE)
            ros.wait_for_service("plan_to_joint_configuration", timeout=self.PATIENCE)
            ros.wait_for_service("get_ik_solutions", timeout=self.PATIENCE)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to UR5 server. Is it running?")
            ros.signal_shutdown("Cannot connect to UR5 server.")
            sys.exit(0)

        self.__service_plan_to_tcp_pose = ros.ServiceProxy(
            "plan_to_tcp_pose", PODService
        )
        self.__service_plan_to_joint_configuration = ros.ServiceProxy(
            "plan_to_joint_configuration", PODService
        )
        self.__service_get_ik_solutions = ros.ServiceProxy(
            "get_ik_solutions", PODService
        )
        self.__service_toppra = ros.ServiceProxy("toppra", PODService)

    def plan_to_tcp_pose(
        self,
        sophie: Optional[np.ndarray] = None,
        wilson: Optional[np.ndarray] = None,
        scene: str = "default",
        min_distance: float | None = None,
        max_distance: float | None = None,
    ) -> Tuple[DualTrajectoryPOD]:
        assert not (sophie is None and wilson is None)

        pod = DualTCPPOD(
            ros.Time.now(),
            tcp_sophie=sophie,
            tcp_wilson=wilson,
            scene=scene,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        response = make_pod_request(
            self.__service_plan_to_tcp_pose, pod, DualTrajectoryPOD
        )
        if response is None:
            raise RuntimeError(
                f"No plan found to tcps: \n\nSophie:\n{sophie}\n\nWilson:\n{wilson}."
            )

        return response

    def plan_to_joint_configuration(
        self,
        sophie: Optional[np.ndarray] = None,
        wilson: Optional[np.ndarray] = None,
        scene: str = "default",
        max_distance: float | None = None,
    ) -> Tuple[DualTrajectoryPOD]:
        assert not (sophie is None and wilson is None)

        pod = DualJointsPOD(
            ros.Time.now(),
            joints_sophie=sophie,
            joints_wilson=wilson,
            scene=scene,
            max_distance=max_distance,
        )

        response = make_pod_request(
            self.__service_plan_to_joint_configuration, pod, DualTrajectoryPOD
        )
        if response is None:
            raise RuntimeError(
                f"No plan found:\nSOPHIE:\n{sophie}\n\nWILSON:\n{wilson}"
            )
        return response

    def get_ik_solutions(
        self,
        sophie: np.ndarray | None = None,
        wilson: np.ndarray | None = None,
        scene: str = "default",
    ):
        pod = DualTCPPOD(
            ros.Time.now(), tcp_sophie=sophie, tcp_wilson=wilson, scene=scene
        )

        response = make_pod_request(self.__service_get_ik_solutions, pod, DualJointsPOD)

        if response is None:
            return np.empty((0,))

        if sophie is None:
            return response.joints_wilson
        elif wilson is None:
            return response.joints_sophie
        else:
            return response.joints_sophie, response.joints_wilson

    def toppra(
        self,
        sophie: np.ndarray | None = None,
        wilson: np.ndarray | None = None,
    ):
        pod = DualJointsPOD(ros.Time.now(), joints_sophie=sophie, joints_wilson=wilson)
        response = make_pod_request(self.__service_toppra, pod, DualTrajectoryPOD)

        if response is None:
            raise RuntimeError(f"No TOPPRA solution found.")

        return response
