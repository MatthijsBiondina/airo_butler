import sys

import numpy as np
from pairo_butler.utils.pods import DualPathPOD, DualTCPPOD, make_pod_request
import rospy as ros
from typing import Any, Callable, Optional
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
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to UR5 server. Is it running?")
            ros.signal_shutdown("Cannot connect to UR5 server.")
            sys.exit(0)

        self.__service_plan_to_tcp_pose = ros.ServiceProxy(
            "plan_to_tcp_pose", PODService
        )

    def plan_to_tcp_pose(
        self, sophie: Optional[np.ndarray] = None, wilson: Optional[np.ndarray] = None
    ):
        assert not (sophie is None and wilson is None)

        pod = DualTCPPOD(ros.Time.now(), tcp_sophie=sophie, tcp_wilson=wilson)
        response = make_pod_request(self.__service_plan_to_tcp_pose, pod, DualPathPOD)

        pyout()
