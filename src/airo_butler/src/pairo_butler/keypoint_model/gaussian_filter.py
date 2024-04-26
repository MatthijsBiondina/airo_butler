from copy import deepcopy
import pickle
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from pairo_butler.utils.pods import ArrayPOD, KeypointUVPOD, publish_pod
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.tools import UGENT, load_config, pyout
from threading import Lock


class GaussianConvolutionFilter:
    RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "gaussian_convolution_filter"):
        self.name = name
        self.config = load_config()
        self.lock = Lock()
        self.rate: ros.Rate

        self.heatmap_sub: ros.Subscriber
        self.buffer: Optional[ArrayPOD] = None

        self.keypoints_uv_pub: ros.Publisher
        self.valid_keypoints_pub: ros.Publisher

        self.heatmap_sigma = self.config.heatmap_sigma
        self.heatmap_threshold = self.config.heatmap_threshold

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.heatmap_sub = ros.Subscriber(
            "/keypoints_heatmap",
            PODMessage,
            self.__sub_callback,
            queue_size=self.QUEUE_SIZE,
        )

        self.keypoints_uv_pub = ros.Publisher(
            "/keypoints_uv", PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.buffer is not None:
                with self.lock:
                    pod = deepcopy(self.buffer)
                    self.buffer = None

                heatmap = self.__preprocess_heatmap(pod.array)
                filtered_heatmap = self.__apply_gaussian_kernel(heatmap)
                x, y = self.__extract_keypoint(filtered_heatmap)

                pod = KeypointUVPOD(
                    pod.timestamp,
                    valid=x is not None and y is not None,
                    x=x,
                    y=y,
                )

                publish_pod(self.keypoints_uv_pub, pod)

            self.rate.sleep()

    def __sub_callback(self, msg):
        pod = pickle.loads(msg.data)
        with self.lock:
            self.buffer = pod

    def __preprocess_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.max(heatmap, axis=0)
        heatmap[heatmap < self.heatmap_threshold] = 0

        return heatmap

    def __apply_gaussian_kernel(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply a Gaussian filter to the heatmap.

        Args:
            heatmap (np.ndarray): The input heatmap array to be filtered.

        Returns:
            np.ndarray: The filtered heatmap.
        """
        # Apply Gaussian filter
        if not isinstance(heatmap, np.ndarray):
            pyout("Error: Input must be a numpy array.")
            return heatmap

        if self.heatmap_sigma <= 0:
            pyout("Error: Sigma must be positive.")
            return heatmap

        filtered_heatmap = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
        return filtered_heatmap

    def __extract_keypoint(self, heatmap: np.ndarray) -> tuple:
        """
        Extract the keypoint (x, y coordinates) from the filtered heatmap by identifying the
        position of the maximum value.

        Args:
            heatmap (np.ndarray): The filtered heatmap from which to extract the keypoint.

        Returns:
            tuple: The (x, y) coordinates of the keypoint.
        """
        if heatmap.size == 0:
            pyout("Error: Heatmap is empty.")
            return None  # Return an invalid keypoint if the heatmap is empty

        # Find the index of the maximum value in the heatmap
        index_of_maximum = np.argmax(heatmap)
        # Convert the 1D index to 2D coordinates
        y, x = np.unravel_index(index_of_maximum, heatmap.shape)

        if heatmap[y, x] > self.heatmap_threshold:
            return (x, y)  # Return coordinates as (x, y)
        else:
            return None, None


def main():
    node = GaussianConvolutionFilter()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
