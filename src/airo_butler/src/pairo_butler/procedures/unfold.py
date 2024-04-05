import sys
from typing import Any, Dict

import numpy as np
from pairo_butler.procedures.subprocedures.kalman_scan import KalmanScan
from pairo_butler.procedures.subprocedures.holdup import Holdup
from pairo_butler.procedures.subprocedures.pickup import Pickup
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.ompl_client import OMPLClient
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout


np.set_printoptions(precision=2, suppress=True)


class UnfoldMachine:
    def __init__(self, name: str = "unfold_machine"):
        self.node_name = name
        self.config = load_config()

        self.ompl: OMPLClient
        self.sophie: UR5eClient
        self.wilson: UR5eClient

        self.kwargs: Dict[str, Any]

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
        # plan = self.ompl.plan_to_joint_configuration(
        #     wilson=np.deg2rad(self.config.joints_shake_wilson),
        #     sophie=np.deg2rad(self.config.joints_shake_sophie)

        # )
        # self.wilson.execute_plan(plan)

        while not ros.is_shutdown():
            ros.loginfo("Startup")
            Startup(**self.kwargs).run()
            ros.loginfo("Pickup")
            while not Pickup(**self.kwargs).run():
                pyout(f"Could not pick up towel. Try again.")

            ros.loginfo("Grasp Corner")
            if Holdup(**self.kwargs).run():
                break

        KalmanScan(**self.kwargs).run()


def main():
    node = UnfoldMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
