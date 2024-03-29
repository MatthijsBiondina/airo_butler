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
import torch
from pairo_butler.kalman_filters.kalman_filter_utils import (
    POD3D,
    KalmanFilterState,
    add_new_measurements_to_state,
    add_remaining_points_as_new_points,
    compute_covariance_over_color,
    compute_covariance_over_position,
    construct_full_covariance_matrix,
    initialize_camera_intrinsics,
    landmark_fusion,
    load_trial,
    plot_state,
    preprocess_measurements,
    scale_hue_within_range,
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
            with torch.no_grad():
                trial: List[POD3D] = load_trial(trial_path)
                preprocess_measurements(trial, self.intrinsics)
                compute_covariance_over_position(trial, self.intrinsics)
                compute_covariance_over_color(trial)
                construct_full_covariance_matrix(trial)

                state = KalmanFilterState(state_size=trial[0].y[0].size)
                for frame in trial:
                    state.Sigma = torch.clamp(state.Sigma, -1e12, 1e12)

                    add_new_measurements_to_state(frame, state)
                    landmark_fusion(state)
                    add_remaining_points_as_new_points(state)
                    scale_hue_within_range(state)
                    plot_state(state)
                    cv2.imshow("RS2", frame.image)
                    cv2.waitKey(10)


def main():
    node = Kalman3DFilter()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
