import pickle
import sys
from typing import List, Optional
import warnings
from PIL import Image
import cv2
import numpy as np
from pairo_butler.camera.zed_camera import ZEDClient
from pairo_butler.plotting.plotting_utils import add_info_to_image
import genpy
from pairo_butler.utils.tools import prog, pyout
from pairo_butler.utils.pods import ZEDPOD
from pairo_butler.plotting.pygame_plotter import PygameWindow
import rospy as ros
from airo_butler.msg import PODMessage
from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_and_visualize_charuco_pose,
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
)


class ZEDStreamRGB:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 30
    SIZE = (405, 720)
    LANDSCAPE = True

    def __init__(self, name: str = "zed_stream") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.zed: Optional[ZEDClient] = None

        self.window = PygameWindow(
            "Zed2i", size=self.SIZE[::-1] if self.LANDSCAPE else self.SIZE
        )

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.zed = ZEDClient()

    def run(self):
        while not ros.is_shutdown():
            frame = (self.zed.pod.rgb_image * 255).astype(np.uint8)
            frame = frame.transpose(1, 0, 2)[::-1]

            # frame = frame[:, :, ::-1]
            frame = cv2.resize(frame, self.SIZE)

            # cv2.imwrite("/home/matt/Pictures/frame.png", frame)

            # intrinsics = self.zed.pod.intrinsics_matrix

            # np.save("/home/matt/Pictures/intrinsics.npy", intrinsics)

            # sys.exit(0)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                detect_and_visualize_charuco_pose(
                    frame,
                    intrinsics=self.zed.pod.intrinsics_matrix,
                    aruco_dict=AIRO_DEFAULT_ARUCO_DICT,
                    charuco_board=AIRO_DEFAULT_CHARUCO_BOARD,
                )

            if self.LANDSCAPE:
                frame = Image.fromarray(frame.transpose(1, 0, 2)[:, ::-1])
                frame = frame.resize(self.SIZE[::-1])
            else:
                frame = Image.fromarray(frame)
                frame = frame.resize(self.SIZE)

            frame = add_info_to_image(
                frame,
                title="Zed2i (RGB)",
                frame_rate=f"{self.zed.fps} Hz",
                latency=f"{self.zed.latency} ms",
            )
            self.window.imshow(frame)
            self.rate.sleep()


def main():
    node = ZEDStreamRGB()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
