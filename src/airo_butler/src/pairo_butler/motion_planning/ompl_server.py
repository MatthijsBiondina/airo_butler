import json
from multiprocessing import Process
import pickle
from typing import Dict, List, Union

import numpy as np
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.ompl_client import OMPLClient
from pairo_butler.procedures.grasp import GraspMachine
from pairo_butler.utils.pods import (
    DualJointsPOD,
    DualTCPPOD,
    DualTrajectoryPOD,
    GraspTCPPOD,
)
from pairo_butler.motion_planning.drake_simulation import DrakeSimulation
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout
from airo_butler.srv import PODService, PODServiceRequest, PODServiceResponse


class OMPL_server:
    PUBLISH_RATE = 30

    def __init__(self, name: str = "ompl_server"):
        self.node_name = name
        self.config = load_config()

        self.simulators: Dict[str, DrakeSimulation] = {}
        self.services: Dict[str, ros.Service] = self.__initialize_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO, anonymous=True)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        try:
            scenes: List[str] = eval(ros.get_param("~scenes", ["default"]))
        except TypeError:
            # todo: debug
            scenes = [
                "default",
                "wilson_holds_charuco",
                "sophie_holds_charuco",
                "hanging_towel",
            ]

        for scene in scenes:
            self.simulators[scene] = DrakeSimulation(scene)

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            for _, simulator in self.simulators.items():
                simulator.update()
            self.rate.sleep()

    def __initialize_services(self):
        self.services = {
            "plan_to_tcp_pose": ros.Service(
                "plan_to_tcp_pose", PODService, self.__plan_to_tcp_pose
            ),
            "plan_to_joint_configuration": ros.Service(
                "plan_to_joint_configuration",
                PODService,
                self.__plan_to_joint_configuration,
            ),
            "get_ik_solutions": ros.Service(
                "get_ik_solutions", PODService, self.__get_ik_solutions
            ),
            "toppra": ros.Service("toppra", PODService, self.__toppra),
        }

    def __plan_to_tcp_pose(self, req: PODServiceRequest):
        input_pod: DualTCPPOD = pickle.loads(req.pod)

        if input_pod.max_distance is None and input_pod.min_distance is None:
            simulator = self.simulators[input_pod.scene]
        else:
            simulator = DrakeSimulation(
                scene_name="default" if input_pod.scene is None else input_pod.scene,
                min_distance=input_pod.min_distance,
                max_distance=input_pod.max_distance,
            )

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

        if input_pod.max_distance is None:
            simulator = self.simulators[input_pod.scene]
        else:
            simulator = DrakeSimulation(
                scene_name="default" if input_pod.scene is None else input_pod.scene,
                max_distance=input_pod.max_distance,
            )
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

    def __get_ik_solutions(self, req: PODServiceRequest):
        input_pod: DualTCPPOD = pickle.loads(req.pod)

        try:
            simulator = self.simulators[input_pod.scene]
            joint_poses_sophie, joint_poses_wilson = simulator.get_ik_solutions(
                input_pod.tcp_sophie, input_pod.tcp_wilson
            )
        except KeyError:
            ros.logwarn(f"Unknown scene: {input_pod.scene}")

        response_pod: DualJointsPOD = DualJointsPOD(
            joints_sophie=joint_poses_sophie,
            joints_wilson=joint_poses_wilson,
            timestamp=input_pod.timestamp,
        )

        response = PODServiceResponse()
        response.pod = pickle.dumps(response_pod)

        return response

    def __toppra(self, req: PODServiceRequest):
        input_pod: DualJointsPOD = pickle.loads(req.pod)

        simulator = self.simulators["default"]
        path_sophie, path_wilson, period = simulator.toppra(
            sophie_path=input_pod.joints_sophie, wilson_path=input_pod.joints_wilson
        )

        response_pod: DualJointsPOD = DualTrajectoryPOD(
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
    node.run()


if __name__ == "__main__":
    main()
