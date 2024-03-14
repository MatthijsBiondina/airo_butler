import pickle
import sys
import time
from typing import Optional

import numpy as np
from pairo_butler.utils.tools import pyout
import rospy as ros
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage


class GraspFirstCorner(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        self.towel_sub: ros.Subscriber = ros.Subscriber(
            "/towel_bot", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.grasp_point: Optional[np.ndarray] = None

    def run(self):
        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie)
        )
        self.sophie.execute_plan(plan)

        while self.grasp_point is None:
            ros.sleep(1 / self.PUBLISH_RATE)

        tcp_approach = np.array(
            [
                [0.0, 0.0, 1.0, -0.2],
                [-1.0, 0.0, 0.0, self.grasp_point[1]],
                [0.0, -1.0, 0.0, self.grasp_point[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_approach, avoid_towel=True)
        grasp_point = np.copy(self.grasp_point)
        self.sophie.execute_plan(plan)

        tcp_grasp = np.array(
            [
                [0.0, 0.0, 1.0, self.grasp_point[0]],
                [-1.0, 0.0, 0.0, self.grasp_point[1]],
                [0.0, -1.0, 0.0, self.grasp_point[2] + 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_grasp)
        self.sophie.execute_plan(plan)
        sys.exit(0)
        self.wilson.close_gripper()

        tcp_hold = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_tcp_pose(wilson=tcp_hold)
        self.wilson.execute_plan(plan)

    def __sub_callback(self, msg):
        pod = pickle.loads(msg.data)
        self.grasp_point = np.array([pod.x, pod.y, pod.z])
