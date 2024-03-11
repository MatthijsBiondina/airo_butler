from multiprocessing import Process
import pickle
from threading import Thread
import time
from typing import Dict, Optional
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import DualTCPPOD
from pairo_butler.motion_planning.ompl_client import OMPLClient
from airo_butler.srv import PODService, PODServiceRequest, PODServiceResponse
import numpy as np
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.drake_simulation import DrakeSimulation
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from pairo_butler.utils.tools import load_config, pyout
import rospy as ros
from pydrake.math import RigidTransform, RollPitchYaw


def request_service():
    time.sleep(2)

    ompl_client = OMPLClient()
    # ompl_client.start_ros()

    transform_0 = RigidTransform(p=[0, 0, 0.35], rpy=RollPitchYaw([-np.pi, 0, 0]))
    tcp_pose_0 = np.ascontiguousarray(transform_0.GetAsMatrix4())

    ompl_client.plan_to_tcp_pose(sophie=tcp_pose_0)


class OMPL_server:
    PUBLISH_RATE = 30

    def __init__(self, name: str = "ompl_server"):
        self.config = load_config()
        self.node_name: str = name
        self.rate: ros.Rate

        self.simulator: DrakeSimulation
        self.planner: DualArmOmplPlanner

        self.services: Dict[str, ros.Service] = self.__initialize_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.simulator = DrakeSimulation()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            self.simulator.update()
            self.rate.sleep()

    def __initialize_services(self):
        services = {
            "plan_to_tcp_pose": ros.Service(
                "plan_to_tcp_pose", PODService, self.__plan_to_tcp_pose
            )
        }
        return services

    def __plan_to_tcp_pose(self, req: PODServiceRequest):
        pod: DualTCPPOD = pickle.loads(req.pod)
        path = self.simulator.plan_to_tcp_pose(pod.tcp_sophie, pod.tcp_wilson)

        pyout()


def main():
    node = OMPL_server()
    node.start_ros()
    Process(target=request_service, daemon=True).start()
    node.run()


if __name__ == "__main__":
    main()
