import numpy as np
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
import rospy as ros


class KalmanScan(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        while not ros.is_shutdown():
            self.sophie.move_to_joint_configuration(
                np.deg2rad(self.config.joints_scan1_sophie)
            )
            self.sophie.move_to_joint_configuration(
                np.deg2rad(self.config.joints_scan2_sophie)
            )
            self.sophie.move_to_joint_configuration(
                np.deg2rad(self.config.joints_scan3_sophie)
            )
            self.sophie.move_to_joint_configuration(
                np.deg2rad(self.config.joints_scan2_sophie)
            )
