import base64
import io
import json
import os
from pathlib import Path
from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
import rospkg
from pairo_butler.kalman_filters.kalman_filter_utils import (
    POD3D,
    compute_Q_matrices,
    initialize_camera_intrinsics,
    load_trial,
    preprocess_measurements,
)
import rospy as ros
import pyrealsense2 as rs
from pairo_butler.utils.tools import listdir, pbar, pyout

np.set_printoptions(precision=2, suppress=True)


class Kalman3DFilter:
    ROOT = "/home/matt/Datasets/pointclouds"

    MIN_DEPTH, MAX_DEPTH = 150, 1000

    def __init__(self, name: str = "kalman_3d_filter"):
        self.name = name
        _, self.intrinsics, _ = initialize_camera_intrinsics()

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)
        ros.loginfo(f"{self.name}: OK!")

    def run(self):
        for trial_path in listdir(self.ROOT):
            trial: List[POD3D] = load_trial(trial_path)
            preprocess_measurements(trial, self.intrinsics)
            compute_Q_matrices(trial, self.intrinsics)


def main():
    node = Kalman3DFilter()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
