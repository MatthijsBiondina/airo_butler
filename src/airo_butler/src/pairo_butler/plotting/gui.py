import os
from typing import Optional

import numpy as np
from pairo_butler.ur3_arms.ur3_constants import *
from pairo_butler.ur3_arms.ur3_client import UR3Client
import rospy as ros


class GUI:
    PUBLISH_RATE: int = 1
    QUEUE_SIZE: int = 2

    def __init__(self, name: str = "airo_butler_gui") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Placeholders
        self.left_arm: Optional[UR3Client] = None
        self.right_arm: Optional[UR3Client] = None

        np.set_printoptions(precision=2, suppress=True, threshold=6, linewidth=80)

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.left_arm = UR3Client("left")
        self.right_arm = UR3Client("right")

    def run(self):
        self.left_arm.move_to_joint_configuration(POSE_LEFT_SCAN)

        self.right_arm.move_to_joint_configuration(POSE_RIGHT_MIDDLE, joint_speed=0.1)
        self.right_arm.move_to_joint_configuration(POSE_RIGHT_CLOCK, joint_speed=0.1)
        self.right_arm.move_to_joint_configuration(POSE_RIGHT_MIDDLE, joint_speed=0.1)
        self.right_arm.move_to_joint_configuration(POSE_RIGHT_COUNTER, joint_speed=0.1)
        self.right_arm.move_to_joint_configuration(POSE_RIGHT_MIDDLE, joint_speed=0.1)

        while True:
            os.system("clear")
            ros.loginfo(f"{self.node_name} - Joint Configurations")

            try:
                left = self.left_arm.get_joint_configuration()

                ros.loginfo(f"Left Arm:  {left / np.pi} * pi.")

                right = self.right_arm.get_joint_configuration()
                ros.loginfo(f"Right Arm: {right / np.pi} * pi.")

            except TimeoutError:
                ros.loginfo("Not yet available")
            self.rate.sleep()


def main():
    gui = GUI()
    gui.start_ros()
    gui.run()


if __name__ == "__main__":
    main()
