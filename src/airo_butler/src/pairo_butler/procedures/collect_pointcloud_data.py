import json
from multiprocessing import Process
import os
import sys
import time

import numpy as np
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
import rospy as ros
from pairo_butler.utils.tools import load_config, makedirs, pyout
from pairo_butler.utils.ros_message_collector import ROSMessageCollector
import cv2


def collection_process():
    node = PointcloudDataCollection()
    node.start_ros()
    node.run()


class PointcloudDataCollection:
    def __init__(self, name="bob"):
        self.name = name
        self.collector: ROSMessageCollector

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)

        self.collector = ROSMessageCollector(
            exact=["/rs2_topic"], approximate=["/ur5e_sophie"]
        )

        ros.loginfo(f"{self.name}: OK!")

    def run(self):
        root = "/home/matt/Datasets/pointclouds"
        os.makedirs(root, exist_ok=True)
        trial = f"{root}/{str(len(os.listdir(root))).zfill(3)}"
        os.makedirs(trial)

        while not ros.is_shutdown():
            package = self.collector.next()

            frame = f"{trial}/{str(len(os.listdir(trial))).zfill(4)}"
            os.makedirs(frame)
            np.save(f"{frame}/color.npy", package["/rs2_topic"].color_frame)
            np.save(f"{frame}/depth.npy", package["/rs2_topic"].depth_frame)
            np.save(f"{frame}/tcp.npy", package["/ur5e_sophie"].tcp_pose)


SOPHIE_CLOCK = np.array([-0.50, -1.0, +0.00, +0.00, -0.35, +0.00]) * np.pi
SOPHIE_MIDDLE = np.array([+0.00, -1.0, +0.50, -0.50, -0.50, +0.00]) * np.pi
SOPHIE_COUNTER = np.array([+0.60, -1.00, +0.25, -0.25, -0.75, +0.00]) * np.pi


def move_arm():
    
    for _ in range(10):
        config = load_config()

        wilson = UR5eClient("wilson")
        sophie = UR5eClient("sophie")

        wilson.move_to_joint_configuration(np.deg2rad(config.joints_hold_wilson))
        sophie.move_to_joint_configuration(np.deg2rad(config.joints_rest_sophie))
        wilson.open_gripper()
        # sys.exit(0)
        sophie.move_to_joint_configuration(SOPHIE_COUNTER)
        wilson.close_gripper()
        time.sleep(3)

        process = Process(target=collection_process, daemon=True)
        process.start()

        sophie.move_to_joint_configuration(SOPHIE_MIDDLE)
        sophie.move_to_joint_configuration(SOPHIE_CLOCK)

        process.kill()

        sophie.move_to_joint_configuration(np.deg2rad(config.joints_rest_sophie))
    wilson.open_gripper()


def main():
    move_arm()


if __name__ == "__main__":
    main()
