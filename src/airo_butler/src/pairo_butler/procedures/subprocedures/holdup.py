import pickle
import sys
import time
from typing import Optional

import numpy as np
from pairo_butler.utils.transformations_3d import (
    homogenous_transformation,
    horizontal_view_rotation_matrix,
)
from pairo_butler.motion_planning.towel_obstacle import TowelObstacle
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.utils.tools import pyout
import rospy as ros
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage


class Holdup(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def run(self):
        try:
            self.__wilson_grab_first_corner()
            self.sophie.open_gripper()
            self.__present()
            if self.towel_on_table():
                self.sophie.open_gripper()
                self.wilson.open_gripper()
                return False
        except RuntimeError:
            return False

        return True

    def __wilson_grab_first_corner(self):
        plan = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_rest_wilson),
            scene="hanging_towel",
        )
        self.wilson.execute_plan(plan)
        ros.sleep(10)

        plan = None
        distance = 0.2
        while plan == None:
            if distance > 0.8:
                raise RuntimeError(f"Could not find approach to towel.")
            for approach_tcp in self.__compute_approach_tcp(
                self.towel_bot.copy(), distance
            ):
                try:
                    R = homogenous_transformation(yaw=-90)
                    approach_tcp = approach_tcp @ R
                    plan = self.ompl.plan_to_tcp_pose(
                        wilson=approach_tcp, scene="hanging_towel"
                    )
                    break
                except RuntimeError:
                    pass
            distance += 0.05

        grasp_point = self.towel_bot.copy()
        self.wilson.execute_plan(plan)

        grasp_tcp = approach_tcp.copy()
        grasp_tcp[:3, 3] = grasp_point + 0.05 * grasp_tcp[:3, 2]

        plan = self.ompl.plan_to_tcp_pose(wilson=grasp_tcp)
        self.wilson.execute_plan(plan)
        self.wilson.close_gripper()

    def __compute_approach_tcp(self, point: np.ndarray, distance: float, steps=10):

        tcps = []

        for z_ in np.linspace(0.1, 0.7, num=steps):
            for theta in np.linspace(0, 90, num=steps):
                theta = np.deg2rad(theta)

                x, y = -np.cos(theta) * distance, np.sin(theta) * distance

                z_axis = np.array([point[0], point[1], point[2]]) - np.array([x, y, z_])
                z_axis /= np.linalg.norm(z_axis)

                H = horizontal_view_rotation_matrix(z_axis)
                H[:3, 3] = np.array([x, y, z_])

                tcps.append(H)
        return tcps

    def __present(self):
        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_shake_sophie),
            wilson=np.deg2rad(self.config.joints_shake_wilson),
        )
        self.wilson.execute_plan(plan)

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie),
            wilson=np.deg2rad(self.config.joints_drop_wilson),
        )
        self.wilson.execute_plan(plan)

        plan = self.ompl.plan_to_joint_configuration(
            # sophie=np.deg2rad(self.config.joints_rest_sophie),
            wilson=np.deg2rad(self.config.joints_hold_wilson),
        )
        self.wilson.execute_plan(plan)
