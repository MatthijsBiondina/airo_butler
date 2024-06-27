import pickle
import sys
import time
from typing import Optional

import numpy as np
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.motion_planning.towel_obstacle import TowelObstacle
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
import rospy as ros
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage


class Pickup(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def run(self):
        self.sophie.open_gripper()
        if self.__pickup_with_sophie():
            return True
        DropTowel(**self.kwargs).run()
        return False

    def __pickup_with_wilson(self):
        time.sleep(2)
        while self.towel_top is None:
            ros.sleep(1 / self.PUBLISH_RATE)

        fails = 0
        while True:
            try:
                grasp_point = self.towel_top
                tcp_pickup = np.array(
                    [
                        [-1.0, 0.0, 0.0, grasp_point[0]],
                        [0.0, 1.0, 0.0, grasp_point[1]],
                        [0.0, 0.0, -1.0, 0.01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )

                plan = self.ompl.plan_to_tcp_pose(wilson=tcp_pickup)
                self.wilson.execute_plan(plan)
                self.wilson.close_gripper()
                break
            except RuntimeError:
                DropTowel(**self.kwargs).run()
                pyout(tcp_pickup)
                fails += 1
                if fails >= 5:
                    return False

        plan = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_hold_wilson)
        )
        self.wilson.execute_plan(plan)
        time.sleep(2)
        if not self.towel_on_table():
            return True
        else:
            self.sophie.open_gripper()
            return False

    def __pickup_with_sophie(self):
        time.sleep(2)
        while self.towel_top is None:
            ros.sleep(1 / self.PUBLISH_RATE)

        fails = 0
        while True:
            try:
                grasp_point = self.towel_top
                tcp_pickup = np.array(
                    [
                        [1.0, 0.0, 0.0, grasp_point[0]],
                        [0.0, -1.0, 0.0, grasp_point[1]],
                        [0.0, 0.0, -1.0, max(grasp_point[2] - 0.06, 0.001)],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )

                plan = self.ompl.plan_to_tcp_pose(sophie=tcp_pickup)
                self.sophie.execute_plan(plan)
                self.sophie.close_gripper()
                break
            except RuntimeError:
                DropTowel(**self.kwargs).run()
                pyout(tcp_pickup)
                fails += 1
                if fails >= 5:
                    return False

        plan = self.ompl.plan_to_tcp_pose(sophie=self.config.tcp_hold_sophie)
        self.sophie.execute_plan(plan)

        # plan = self.ompl.plan_to_joint_configuration(
        #     sophie=np.deg2rad(self.config.joints_hold_sophie)
        # )
        # self.sophie.execute_plan(plan)

        time.sleep(2)

        if not self.towel_on_table():
            return True
        else:
            self.sophie.open_gripper()
            return False
