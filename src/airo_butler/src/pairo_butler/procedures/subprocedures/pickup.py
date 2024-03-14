import pickle
import time
from typing import Optional

import numpy as np
from pairo_butler.utils.tools import pyout
import rospy as ros
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage


class Pickup(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        self.towel_sub: ros.Subscriber = ros.Subscriber(
            "/towel_top", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.towel_top: Optional[np.ndarray] = None

    def run(self):
        while self.towel_top is None:
            ros.sleep(1 / self.PUBLISH_RATE)

        tcp_pickup = np.array(
            [
                [-1.0, 0.0, 0.0, self.towel_top[0]],
                [0.0, 1.0, 0.0, self.towel_top[1]],
                [0.0, 0.0, -1.0, max(0.01, self.towel_top[2] - 0.1)],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        plan = self.ompl.plan_to_tcp_pose(wilson=tcp_pickup)
        self.wilson.execute_plan(plan)
        self.wilson.close_gripper()

        tcp_hold = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        plan = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_hold_wilson)
        )
        self.wilson.execute_plan(plan)

    def __sub_callback(self, msg):
        pod = pickle.loads(msg.data)
        self.towel_top = np.array([pod.x, pod.y, pod.z])
