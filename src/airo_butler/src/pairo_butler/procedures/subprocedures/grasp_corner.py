import pickle
import numpy as np
from pairo_butler.utils.transformations_3d import homogenous_transformation
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage
import rospy as ros


class GraspCorner(Subprocedure):
    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)

        self.state = state

    def run(self):
        corners = self.__choose_corner()

        for corner in corners:
            approach_tcps, grasp_tcps = self.__compute_approach(corner)

            for approach_tcp, grasp_tcp in zip(approach_tcps, grasp_tcps):
                try:
                    plan = self.ompl.plan_to_tcp_pose(
                        sophie=approach_tcp, scene="hanging_towel"
                    )
                    self.sophie.execute_plan(plan)
                except RuntimeError:
                    continue

                try:
                    plan = self.ompl.plan_to_tcp_pose(sophie=grasp_tcp)
                    self.sophie.execute_plan(plan)
                except RuntimeError:
                    plan = self.ompl.plan_to_tcp_pose(sophie=approach_tcp)
                    self.sophie.execute_plan(plan)
                    continue

                self.sophie.close_gripper()
                return True
        return False

    def __choose_corner(self):
        covariances = self.state.covariances
        certainty = np.linalg.det(covariances)

        indexes_by_certainty = np.argsort(certainty)
        if indexes_by_certainty.size > 2:
            indexes_by_certainty = indexes_by_certainty[:2]

        return self.state.means[indexes_by_certainty]

    def __compute_approach(self, kp):
        kp_tcp = np.array(
            [
                [np.sin(kp[3]), 0.0, np.cos(kp[3]), kp[0]],
                [-np.cos(kp[3]), 0.0, np.sin(kp[3]), kp[1]],
                [0.0, -1.0, 0.0, kp[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        grasp_tcp_origin = kp_tcp @ homogenous_transformation(pitch=180)

        pyout(grasp_tcp_origin)

        grasp_poses = []
        approach_poses = []
        for d_approach in np.linspace(0, 45, num=100):
            grasp_tcp = grasp_tcp_origin @ homogenous_transformation(
                roll=np.random.uniform(-d_approach, d_approach),
                pitch=np.random.uniform(-d_approach, d_approach),
                # roll=np.random.uniform(-d_approach, d_approach),
            )
            approach_tcp = grasp_tcp.copy()

            grasp_tcp[:3, 3] += grasp_tcp[:3, 2] * 0.05
            approach_tcp[:3, 3] -= approach_tcp[:3, 2] * 0.15

            for _ in range(50):
                if np.linalg.norm(approach_tcp[:2, 3]) > 0.3:
                    break
                approach_tcp[:3, 3] -= approach_tcp[:3, 2] * 0.01

            grasp_poses.append(grasp_tcp)
            approach_poses.append(approach_tcp)

        return approach_poses, grasp_poses
