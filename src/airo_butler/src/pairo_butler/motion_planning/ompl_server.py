from multiprocessing import Process
import pickle
from threading import Thread
import time
from typing import Dict, Optional
from airo_butler.msg import PODMessage
from pairo_butler.procedures import unfold
from pairo_butler.motion_planning.towel_obstacle import TowelObstacle
from pairo_butler.utils.pods import (
    DualJointsPOD,
    DualTCPPOD,
    DualTrajectoryPOD,
    TowelPOD,
)
from pairo_butler.motion_planning.ompl_client import OMPLClient
from airo_butler.srv import PODService, PODServiceRequest, PODServiceResponse
import numpy as np
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.drake_simulation import DrakeSimulation
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from pairo_butler.utils.tools import load_config, pyout
import rospy as ros
from pydrake.math import RigidTransform, RollPitchYaw


np.set_printoptions(precision=0, suppress=True)


def request_process():
    unfold.main()


class OMPL_server:
    PUBLISH_RATE = 30

    def __init__(self, name: str = "ompl_server"):
        self.config = load_config()
        self.node_name: str = name
        self.rate: ros.Rate

        self.simulator_without_towel: DrakeSimulation
        self.simulator_with_towel: DrakeSimulation

        self.planner_without_towel: DualArmOmplPlanner
        self.planner_with_towel: DualArmOmplPlanner

        self.services: Dict[ros.Service] = self.__initialize_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.simulator_without_towel = DrakeSimulation()
        self.simulator_with_towel = DrakeSimulation(TowelObstacle())

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            try:
                self.simulator_without_towel.update()
                self.simulator_with_towel.update()
            except AttributeError:
                pass
            self.rate.sleep()

    def __initialize_services(self):
        services = {
            "plan_to_tcp_pose": ros.Service(
                "plan_to_tcp_pose", PODService, self.__plan_to_tcp_pose
            ),
            "plan_to_joint_configuration": ros.Service(
                "plan_to_joint_configuration",
                PODService,
                self.__plan_to_joint_configuration,
            ),
        }
        return services

    def __plan_to_tcp_pose(self, req: PODServiceRequest):
        input_pod: DualTCPPOD = pickle.loads(req.pod)
        if input_pod.avoid_towel:
            simulator = self.simulator_with_towel
        else:
            simulator = self.simulator_without_towel
        path_sophie, path_wilson, period = simulator.plan_to_tcp_pose(
            input_pod.tcp_sophie, input_pod.tcp_wilson
        )
        response_pod: DualTrajectoryPOD = DualTrajectoryPOD(
            timestamp=input_pod.timestamp,
            path_sophie=path_sophie,
            path_wilson=path_wilson,
            period=period,
        )

        response = PODServiceResponse()
        response.pod = pickle.dumps(response_pod)
        return response

    def __plan_to_joint_configuration(self, req: PODServiceRequest):
        input_pod: DualJointsPOD = pickle.loads(req.pod)
        if input_pod.avoid_towel:
            simulator = self.simulator_with_towel
        else:
            simulator = self.simulator_without_towel

        path_sophie, path_wilson, period = simulator.plan_to_joint_configuration(
            input_pod.joints_sophie, input_pod.joints_wilson
        )
        response_pod: DualTrajectoryPOD = DualTrajectoryPOD(
            timestamp=input_pod.timestamp,
            path_sophie=path_sophie,
            path_wilson=path_wilson,
            period=period,
        )

        response = PODServiceResponse()
        response.pod = pickle.dumps(response_pod)
        return response


def main():
    node = OMPL_server()
    node.start_ros()
    Process(target=request_process, daemon=True).start()

    node.run()


if __name__ == "__main__":
    main()
