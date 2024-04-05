import pickle
import sys

import numpy as np
from pairo_butler.motion_planning.towel_obstacle import TowelObstacle
from pairo_butler.utils.pods import (
    BooleanPOD,
    DualJointsPOD,
    DualTrajectoryPOD,
    DualTCPPOD,
    TowelPOD,
    make_pod_request,
)
import rospy as ros
from typing import Any, Callable, Optional, Tuple
from pairo_butler.utils.tools import load_config, pyout
from airo_butler.srv import PODService, PODServiceResponse
from airo_butler.msg import PODMessage


class OMPLClient:
    QUEUE_SIZE = 2
    PATIENCE = 5.0

    def __init__(self, name: str = "ompl_client"):
        self.name = name
        self.config = load_config()

        self.__service_plan_to_tcp_pose: Callable[[Any], Any]
        self.__wait_for_services()

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)

    def __wait_for_services(self):
        try:
            ros.wait_for_service("plan_to_tcp_pose", timeout=self.PATIENCE)
            ros.wait_for_service("plan_to_joint_configuration", timeout=self.PATIENCE)
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

    def plan_to_tcp_pose(
        self,
        sophie: Optional[np.ndarray] = None,
        wilson: Optional[np.ndarray] = None,
        scene: str = "default",
    ) -> Tuple[DualTrajectoryPOD]:
        assert not (sophie is None and wilson is None)

        pod = DualTCPPOD(
            ros.Time.now(),
            tcp_sophie=sophie,
            tcp_wilson=wilson,
            scene=scene,
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
    ) -> Tuple[DualTrajectoryPOD]:
        assert not (sophie is None and wilson is None)

        pod = DualJointsPOD(
            ros.Time.now(),
            joints_sophie=sophie,
            joints_wilson=wilson,
            scene=scene,
        )

        response = None
        for _ in range(3):
            if response is not None:
                return response
            response = make_pod_request(
                self.__service_plan_to_joint_configuration, pod, DualTrajectoryPOD
            )

        pyout(sophie)
        pyout(wilson)
        raise RuntimeError("No plan found.")
