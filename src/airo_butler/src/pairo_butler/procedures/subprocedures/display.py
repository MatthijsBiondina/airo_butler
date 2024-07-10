import pickle
import numpy as np
from pairo_butler.utils.transformations_3d import homogenous_transformation
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage
import rospy as ros


class DisplayTowel(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        edge_length = np.linalg.norm(
            self.wilson.get_tcp_pose()[:3, 3] - self.sophie.get_tcp_pose()[:3, 3]
        )

        try:

            tcp_wilson = np.array(
                [
                    [-1.0, 0.0, 0.0, -0.15],
                    [0.0, 0.0, -1.0, edge_length / 2 - 0.01],
                    [0.0, -1.0, 0.0, 0.75],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            tcp_sophie = np.array(
                [
                    [-1.0, 0.0, 0.0, -0.15],
                    [0.0, 0.0, 1.0, -edge_length / 2 + 0.01],
                    [0.0, 1.0, 0.0, 0.75],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            plan = self.ompl.plan_to_tcp_pose(
                sophie=tcp_sophie,
                wilson=tcp_wilson,
                min_distance=edge_length - 0.20,
                max_distance=edge_length + 0.04,
            )
            self.sophie.execute_plan(plan)

            return True
        except RuntimeError:
            return False
