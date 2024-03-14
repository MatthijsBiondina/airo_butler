from typing import Any, Dict

import numpy as np
from pairo_butler.procedures.subprocedures.grasp_first_corner import GraspFirstCorner
from pairo_butler.procedures.subprocedures.pickup import Pickup
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.motion_planning.ompl_client import OMPLClient
import rospy as ros
from pairo_butler.utils.tools import load_config


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
        try:
            ros.init_node(self.node_name, log_level=ros.INFO)
        except ros.exceptions.ROSException:
            pass

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
        # Pickup(**self.kwargs).run()
        GraspFirstCorner(**self.kwargs).run()


def main():
    node = UnfoldMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()


# def request_service():
#     SOPHIE_SLEEP = np.array([+0.00, -1, +0.50, -0.50, -0.50, +0.00]) * np.pi
#     WILSON_SLEEP = np.array([+0.00, -0, -0.50, -0.50, +0.50, +0.00]) * np.pi

#     sophie = UR5eClient("sophie")

#     ompl_client = OMPLClient()

#     plan = ompl_client.plan_to_joint_configuration(
#         sophie=SOPHIE_SLEEP,
#         wilson=WILSON_SLEEP,
#     )
#     sophie.execute_plan(plan)

#     # transform_0 = RigidTransform(p=[0, 0, 0.35], rpy=RollPitchYaw([-np.pi, 0, 0]))
#     # tcp_pose_0 = np.ascontiguousarray(transform_0.GetAsMatrix4())
#     # plan = ompl_client.plan_to_tcp_pose(sophie=tcp_pose_0)
#     # sophie.execute_plan(plan)

#     pyout("Done!")
