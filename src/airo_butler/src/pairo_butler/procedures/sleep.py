import numpy as np
from pairo_butler.procedures.subprocedures.goodnight import Goodnight
from pairo_butler.procedures.subprocedures.startup import Startup
import rospy as ros
from typing import Any, Dict
from pairo_butler.motion_planning.ompl_client import OMPLClient
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.utils.tools import load_config


class StartupMachine:
    def __init__(self, name: str = "startup_machine"):
        self.node_name = name
        self.config = load_config()

        self.ompl: OMPLClient
        self.sophie: UR5eClient

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
        # Startup(**self.kwargs).run()
        # self.sophie.close_gripper()
        # Goodnight(**self.kwargs).run()

        pose = np.array(
            [
                [-1.0, 0.0, 0.0, -0.5],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_tcp_pose(wilson=pose)
        self.wilson.execute_plan(plan)


def main():
    node = StartupMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
