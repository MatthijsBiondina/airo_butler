import math
import sys
import time
from typing import Any, Dict

import numpy as np
from pairo_butler.procedures.subprocedures.goodnight import Goodnight
from pairo_butler.data.data_collector import DataCollector
from pairo_butler.procedures.subprocedures.kalman_scan import KalmanScan
from pairo_butler.procedures.subprocedures.holdup import Holdup
from pairo_butler.procedures.subprocedures.pickup import Pickup
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.ompl_client import OMPLClient
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout

from pairo_butler.utils.tools import load_config


class CollectDataProcedure:
    NR_OF_TRIALS = 10
    NR_OF_TOWELS = 1

    def __init__(self, name: str = "collect_data_procedure"):
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
        t_start = time.time()
        trial_nr = 0
        while time.time() < t_start + 60 * 60:
            trial_nr += 1

            ros.loginfo(f"TRIAL: {trial_nr}")

            while not ros.is_shutdown():
                ros.loginfo("Startup")
                Startup(**self.kwargs).run()
                ros.loginfo("Pickup")
                while not Pickup(**self.kwargs).run():
                    pyout(f"Could not pick up towel. Try again.")

                ros.loginfo("Grasp Corner")
                if Holdup(**self.kwargs).run():
                    break

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_scan1_sophie),
                scene="hanging_towel",
            )
            self.sophie.execute_plan(plan)

            ros.sleep(10)
            DataCollector.start_recording()
            KalmanScan(**self.kwargs).run()
            DataCollector.pause_recording()
            DataCollector.save_recording()

        Startup(**self.kwargs).run()
        Goodnight(**self.kwargs).run()


def main():
    node = CollectDataProcedure()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
