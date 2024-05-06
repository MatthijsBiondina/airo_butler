import sys
import time
from typing import Any, Dict

import numpy as np
from pairo_butler.utils.pods import KalmanFilterStatePOD
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


class GraspMachine:
    def __init__(self, name: str = "grasp_machine"):
        self.node_name = name
        self.config = load_config()

        self.ompl: OMPLClient
        self.sophie: UR5eClient
        self.wilson: UR5eClient

        self.kwargs: Dict[str, Any]

        self.state_listener = KalmanFilterClient()

    def start_ros(self):

        ros.init_node(self.node_name, log_level=ros.INFO)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        self.ompl = OMPLClient()
        self.sophie = UR5eClient("sophie")
        self.wilson = UR5eClient("wilson")

        self.kwargs = {
            "sophie": self.sophie,
            "wilson": self.wilson,
            "ompl": self.ompl,
            "config": self.config,
        }

        while not ros.is_shutdown():
            while not ros.is_shutdown():
                Startup(**self.kwargs).run()
                while not Pickup(**self.kwargs).run():
                    pass
                if Holdup(**self.kwargs).run():
                    break
            state = KalmanScan(**self.kwargs).run()
            np.save(f"{self.config.res_dir}/mean.npy", state.means)
            np.save(f"{self.config.res_dir}/cov.npy", state.covariances)

            state = KalmanFilterStatePOD(
                means=np.load(f"{self.config.res_dir}/mean.npy"),
                covariances=np.load(f"{self.config.res_dir}/cov.npy"),
                timestamp=ros.Time.now(),
                camera_tcp=None,
            )
            if GraspCorner(state, **self.kwargs).run():
                pyout("Grasped!")
                # DisplayTowel(**self.kwargs).run()
                # ros.sleep(5)
                break

        #     plan = self.ompl.plan_to_joint_configuration(
        #         sophie=np.deg2rad(self.config.joints_scan1_sophie),
        #         scene="hanging_towel",
        #     )
        #     self.sophie.execute_plan(plan)
        #     ros.sleep(10)

        #     KalmanScan(**self.kwargs).run()
        #     grasped = GraspCorner(self.state_listener.state, **self.kwargs).run()

        # RS2Recorder.stop()
        # RS2Recorder.save()

        Goodnight(**self.kwargs).run()


def main():
    node = GraspMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
