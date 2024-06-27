import pickle
import sys
import time
from typing import Any, Dict

import numpy as np
from pairo_butler.procedures.subprocedures.record_towel_surface import (
    RecordTowelSurface,
)
from pairo_butler.camera.rs2_recorder import RS2Recorder
from pairo_butler.procedures.subprocedures.display import DisplayTowel
from pairo_butler.kalman_filters.kalman_filter import KalmanFilterClient
from pairo_butler.procedures.subprocedures.grasp_corner import GraspCorner
from pairo_butler.procedures.subprocedures.goodnight import Goodnight
from pairo_butler.procedures.subprocedures.fling import Fling
from pairo_butler.procedures.subprocedures.kalman_scan import KalmanScan
from pairo_butler.procedures.subprocedures.holdup import Holdup
from pairo_butler.procedures.subprocedures.pickup import Pickup
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.ompl_client import OMPLClient
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout


np.set_printoptions(precision=3, suppress=True)


class UnfoldMachine:
    def __init__(self, name: str = "unfold_machine"):
        self.node_name = name
        self.config = load_config()

        self.ompl: OMPLClient
        self.sophie: UR5eClient
        self.wilson: UR5eClient

        self.kwargs: Dict[str, Any]

        self.state_listener = KalmanFilterClient()

    def start_ros(self):

        ros.init_node(self.node_name, log_level=ros.INFO)

        self.ompl = OMPLClient()
        self.sophie = UR5eClient("sophie")
        self.wilson = UR5eClient("wilson")

        self.kwargs = {
            "sophie": self.sophie,
            "wilson": self.wilson,
            "ompl": self.ompl,
            "config": self.config,
        }

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        self.pickup()
        self.give_back_towel()
        self.recieve_new_towel()
        self.display()
        RecordTowelSurface(**self.kwargs).run()
        Startup(**self.kwargs).run()
        Goodnight(**self.kwargs).run()

    def pickup(self):
        Startup(**self.kwargs).run()
        while not Pickup(**self.kwargs).run():
            pass

    def give_back_towel(self):
        tcp = np.array(
            [
                [-1.0, 0.0, 0.0, -0.15],
                [0.0, 0.0, -1.0, -1.0],
                [0.0, -1.0, 0.0, 0.9],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_tcp_pose(sophie=tcp)
        self.sophie.execute_plan(plan)
        self.sophie.open_gripper()

    def recieve_new_towel(self):
        self.sophie.execute_plan(
            self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad([90, -180, 90, -90, -90, 0])
            )
        )

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(
                np.array([38.751, -161.159, 111.282, 49.882, 38.749, 0.001])
            ),
            wilson=np.deg2rad(
                np.array([-12.145, -176.531, -16.415, 12.95, 102.14, 0.0])
            ),
        )
        self.wilson.execute_plan(plan)
        ros.sleep(5)
        self.wilson.close_gripper()
        ros.sleep(10)
        self.sophie.close_gripper()

    def display(self):
        with open("./plan.pkl", "rb") as f:
            plan = pickle.load(f)

        self.sophie.execute_plan(plan)

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(
                np.array([-35.649, -148.26, 105.914, -137.656, -54.348, 180])
            ),
            max_distance=0.45,
        )
        self.sophie.execute_plan(plan)

        tcp_sophie = self.sophie.get_tcp_pose()
        tcp_wilson = self.wilson.get_tcp_pose()

        cmd = ""
        while not cmd == "n":
            cmd = input("Move effectors (+/-/n)? ")

            if cmd == "+":
                tcp_sophie[1, -1] -= 0.025
                tcp_wilson[1, -1] += 0.025
            elif cmd == "-":
                tcp_sophie[1, -1] += 0.025
                tcp_wilson[1, -1] -= 0.025
            elif cmd == "n":
                break
            else:
                continue

            try:
                plan = self.ompl.plan_to_tcp_pose(
                    sophie=tcp_sophie,
                    wilson=tcp_wilson,
                    max_distance=max(
                        self.wilson.get_tcp_pose()[1, -1]
                        - self.sophie.get_tcp_pose()[1, -1]
                        + 0.05,
                        tcp_wilson[1, -1] - tcp_sophie[1, -1] + 0.05,
                    ),
                )
                self.sophie.execute_plan(plan)
            except RuntimeError:
                ros.logwarn(f"Couldn't find plan.")


def main():
    node = UnfoldMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
