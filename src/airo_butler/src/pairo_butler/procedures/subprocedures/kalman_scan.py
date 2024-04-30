import numpy as np
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
import rospy as ros


class KalmanScan(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        if not np.all(
            np.isclose(
                np.deg2rad(self.config.joints_scan1_sophie),
                self.sophie.get_joint_configuration(),
                atol=1e-1,
            )
        ):
            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_scan1_sophie),
                scene="hanging_towel",
            )
            self.sophie.execute_plan(plan)
            ros.sleep(0.5)

        KalmanFilter.reset()

        self.sophie.move_to_joint_configuration(
            np.deg2rad(self.config.joints_scan2_sophie), joint_speed=0.15
        )
        self.sophie.move_to_joint_configuration(
            np.deg2rad(self.config.joints_scan3_sophie), joint_speed=0.15
        )
