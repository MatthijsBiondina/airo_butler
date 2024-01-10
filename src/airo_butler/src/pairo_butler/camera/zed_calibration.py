import pickle
import time
from typing import Optional

import numpy as np
from pairo_butler.utils.tools import pyout
from pairo_butler.ur3_arms.ur3_client import UR3Client
from pairo_butler.utils.pods import ZEDPOD
import rospy as ros
from airo_butler.msg import PODMessage


# Arm poses
RARM_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
LARM_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi

# States
STARTUP_STATE = 0
GRAB_STATE = 1
CALIBRATION_STATE = 2


class ZEDCalibration:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    """
    State machine for calibrating eye-to-hand zed2i camera
    """

    def __init__(self, name: str = "zed_calibration") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.subscriber: Optional[ros.Subscriber] = None
        self.left_arm: Optional[UR3Client] = None
        self.right_arm: Optional[UR3Client] = None

        # Placeholders:
        self.frame: Optional[np.ndarray] = None

        self.state: str = STARTUP_STATE

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.subscriber = ros.Subscriber(
            "/zed2i", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.left_arm = UR3Client("left")
        self.right_arm = UR3Client("right")

    def __sub_callback(self, msg: PODMessage):
        pod: ZEDPOD = pickle.loads(msg.data)
        self.frame = pod.rgb_image

    def run(self):
        while not ros.is_shutdown():
            if self.state == STARTUP_STATE:
                self.state = self.__startup()
            elif self.state == GRAB_STATE:
                self.state = self.__grab()
            self.rate.sleep()

    def __startup(self):
        self.right_arm.move_to_joint_configuration(RARM_REST)
        self.left_arm.move_to_joint_configuration(LARM_REST)
        self.left_arm.close_gripper()
        return GRAB_STATE

    def __grab(self):
        while not ros.is_shutdown() and self.left_arm.get_gripper_width() < 0.002:
            self.left_arm.open_gripper()
            time.sleep(1)
            self.left_arm.close_gripper()
            time.sleep(1)
        time.sleep(5)

        return CALIBRATION_STATE


def main():
    node = ZEDCalibration()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
