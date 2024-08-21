import sys
import time
from typing import Any, Dict

import numpy as np
from pairo_butler.data.data_collector import DataCollector
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


class CollectDataMachine:
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
        towel_name = "BIG_WHITE"
        for trial in range(10):
            ros.loginfo(f"TRIAL: {towel_name}_{trial}")

            Startup(**self.kwargs).run()
            pickup_success = False
            while not pickup_success:
                Startup(**self.kwargs).run()
                while not Pickup(**self.kwargs).run():
                    pass
                pickup_success = Holdup(**self.kwargs).run()

            self.sophie.open_gripper()

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_rest_sophie),
                wilson=np.deg2rad(self.config.joints_hold_wilson),
                # wilson=None,
            )
            self.wilson.execute_plan(plan)

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_scan1_sophie),
                scene="hanging_towel",
            )
            self.sophie.execute_plan(plan)
            ros.sleep(0.5)

            DataCollector.start_recording()
            KalmanScan(**self.kwargs).run()
            DataCollector.pause_recording()
            DataCollector.save_recording(
                f"{self.config.data_folder}/{towel_name}_{trial}A"
            )

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_scan1_sophie),
                scene="hanging_towel",
            )
            self.sophie.execute_plan(plan)
            ros.sleep(0.5)

            DataCollector.start_recording()
            KalmanScan(**self.kwargs).run(flipped=True)
            DataCollector.pause_recording()
            DataCollector.save_recording(
                f"{self.config.data_folder}/{towel_name}_{trial}B"
            )

        Goodnight(**self.kwargs).run()


if __name__ == "__main__":
    node = CollectDataMachine()
    node.start_ros()
    node.run()
