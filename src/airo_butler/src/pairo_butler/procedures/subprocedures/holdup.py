import pickle
import sys
import time
from typing import Optional

import numpy as np
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
        if self.wilson.gripper_closed():
            n = 2
        else:
            n = 1

        for _ in range(n):
            if self.wilson.gripper_closed():
                self.__sophie_grasp_lowest_point()
                self.wilson.open_gripper()
            elif self.sophie.gripper_closed():
                self.__wilson_grab_first_corner()
                self.sophie.open_gripper()
            self.__present()
            time.sleep(2)
            if self.towel_on_table():
                self.sophie.open_gripper()
                self.wilson.open_gripper()
                return False

        return True

    def __sophie_grasp_lowest_point(self):
        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie),
            wilson=np.deg2rad(self.config.joints_hold_wilson),
        )
        self.sophie.execute_plan(plan)
        time.sleep(5)

        grasp_point = np.copy(self.towel_bot)
        pyout(grasp_point)
        approach_from_front = grasp_point[0] < 0
        pyout(approach_from_front)
        try:
            for approach_tcp in self.__compute_approach_tcp(
                grasp_point, approach_from_front=approach_from_front
            ):
                try:
                    radius = TowelObstacle().radius + 0.01
                    while (
                        -radius < approach_tcp[0, 3] < radius
                        and -radius < approach_tcp[1, 3] < radius
                    ):
                        approach_tcp[:2, 3] -= 0.01 * approach_tcp[:2, 2]
                    while np.linalg.norm(approach_tcp[:2, 3] - grasp_point[:2]) < 0.1:
                        approach_tcp[:2, 3] -= 0.01 * approach_tcp[:2, 2]

                    plan = self.ompl.plan_to_tcp_pose(
                        sophie=approach_tcp, avoid_towel=False
                    )
                except RuntimeError:
                    pyout(f"\nInvalid:\n{approach_tcp}")
                    continue
                raise BreakException()
        except BreakException:
            pyout(f"\nSolution Found:\n{approach_tcp}")
            pass  # expected behavior. Valid plan found.
        else:
            approach_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.35],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            plan = self.ompl.plan_to_tcp_pose(sophie=approach_tcp)

        grasp_point = np.copy(self.towel_bot).tolist()
        self.sophie.execute_plan(plan)

        grasp_tcp = np.copy(approach_tcp)
        grasp_tcp[:3, 3] = grasp_point
        grasp_tcp[:2, 3] += 0.05 * grasp_tcp[:2, 2]

        plan = self.ompl.plan_to_tcp_pose(sophie=grasp_tcp)
        self.sophie.execute_plan(plan)
        self.sophie.close_gripper()

    def __wilson_grab_first_corner(self):
        plan = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_rest_wilson),
            sophie=np.deg2rad(self.config.joints_hold_sophie),
        )
        self.wilson.execute_plan(plan)
        time.sleep(5)

        approach_x, approach_y, approach_z = np.copy(self.towel_bot).tolist()
        approach_from_front = approach_x < 0
        approach_x = -0.2 if approach_from_front else 0.2
        try:
            tcp_approach = self.__compute_approach_tcp(
                np.array([approach_x, approach_y, approach_z]),
                approach_from_front=approach_from_front,
            )
            plan = self.ompl.plan_to_tcp_pose(wilson=tcp_approach, avoid_towel=True)
        except RuntimeError:
            tcp_approach = self.__compute_approach_tcp(
                np.array([approach_x, approach_y, approach_z]),
                approach_from_front=approach_from_front,
                flipped=True,
            )
            plan = self.ompl.plan_to_tcp_pose(wilson=tcp_approach, avoid_towel=True)

        grasp_x, grasp_y, grasp_z = np.copy(self.towel_bot).tolist()
        self.wilson.execute_plan(plan)

        grasp_x = grasp_x + (0.05 if approach_from_front else -0.05)
        tcp_grasp = self.__compute_approach_tcp(
            np.array([grasp_x, grasp_y, grasp_z]),
            approach_from_front=approach_from_front,
        )
        plan = self.ompl.plan_to_tcp_pose(wilson=tcp_grasp)
        self.wilson.execute_plan(plan)
        self.wilson.close_gripper()

    def __compute_approach_tcp(self, point, approach_from_front=True, flipped=False):
        tcps = [
            np.array(
                [
                    [0.0, 0.0, 1.0, point[0]],
                    [-1.0, 0.0, 0.0, point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [np.sqrt(2), 0.0, np.sqrt(2), point[0]],
                    [-np.sqrt(2), 0.0, np.sqrt(2), point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [-np.sqrt(2), 0.0, np.sqrt(2), point[0]],
                    [-np.sqrt(2), 0.0, -np.sqrt(2), point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0.0, 0.0, point[0]],
                    [0.0, 0.0, 1.0, point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [-1.0, 0.0, 0.0, point[0]],
                    [0.0, 0.0, -1.0, point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [np.sqrt(2), 0.0, -np.sqrt(2), point[0]],
                    [np.sqrt(2), 0.0, np.sqrt(2), point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [-np.sqrt(2), 0.0, -np.sqrt(2), point[0]],
                    [np.sqrt(2), 0.0, -np.sqrt(2), point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, -1.0, point[0]],
                    [1.0, 0.0, 0.0, point[1]],
                    [0.0, -1.0, 0.0, point[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]

        if approach_from_front:
            return tcps
        else:
            return tcps[::-1]

    def __present(self):
        if self.wilson.gripper_closed():
            plan = self.ompl.plan_to_tcp_pose(
                sophie=self.config.tcp_rest_sophie, wilson=self.config.tcp_drop
            )
            self.wilson.execute_plan(plan)

            plan = self.ompl.plan_to_joint_configuration(
                wilson=np.deg2rad(self.config.joints_rest_wilson)
            )
            self.wilson.execute_plan(plan)

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_rest_sophie),
                wilson=np.deg2rad(self.config.joints_hold_wilson),
            )
            self.wilson.execute_plan(plan)
        elif self.sophie.gripper_closed():
            plan = self.ompl.plan_to_tcp_pose(
                sophie=self.config.tcp_drop, wilson=self.config.tcp_rest_wilson
            )
            self.wilson.execute_plan(plan)

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_rest_sophie)
            )
            self.sophie.execute_plan(plan)

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_hold_sophie),
                wilson=np.deg2rad(self.config.joints_rest_wilson),
            )
            self.wilson.execute_plan(plan)
