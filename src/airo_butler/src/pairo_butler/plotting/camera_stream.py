import warnings
import pygame
import pickle
from typing import List, Optional
import PIL
from PIL import Image
import numpy as np
from pairo_butler.camera.rs2_camera import RS2Client
from pairo_butler.plotting.pygame_plotter import PygameWindow
import genpy
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import ImagePOD
from pairo_butler.plotting.plotting_utils import add_info_to_image

from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_and_visualize_charuco_pose,
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
)


class CameraStream:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 30

    def __init__(self, name: str = "camera_stream"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        self.rs2: Optional[RS2Client] = None

        # Placeholders
        self.frame: Optional[PIL.Image] = None
        self.intrinsics_matrix: Optional[np.ndarray] = None
        self.frame_timestamp: Optional[ros.Time] = None
        self.timestamps: List[ros.Time] = []

        # Pygame initialization
        self.window = PygameWindow("Realsense2 (RGB)", (1024, 512))

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.rs2 = RS2Client()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():

            # image = np.copy(np.array(self.rs2.pod.image))

            color_frame = np.copy(np.array(self.rs2.pod.color_frame))
            depth_frame = np.copy(np.array(self.rs2.pod.depth_frame))

            image = np.concatenate(
                (color_frame, np.stack((depth_frame,) * 3, axis=-1)), axis=1
            )

            image = Image.fromarray(image)
            image = add_info_to_image(
                image,
                title="RealSense2 (RGB)",
                frame_rate=f"{self.rs2.fps} Hz",
                latency=f"{self.rs2.latency} ms",
            )
            self.window.imshow(image)
            self.rate.sleep()


def main():
    node = CameraStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
