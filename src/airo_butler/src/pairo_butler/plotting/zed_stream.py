import pickle
from typing import List, Optional
from PIL import Image
import cv2
import numpy as np
from pairo_butler.plotting.plotting_utils import add_info_to_image
import genpy
from pairo_butler.utils.tools import pyout
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
    SIZE = (1280, 720)

    def __init__(self, name: str = "zed_stream") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.subscriber: Optional[ros.Subscriber] = None

        # Placeholders
        self.depth_image: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None
        self.point_cloud: Optional[np.ndarray] = None
        self.rgb_image: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.timestamps: List[ros.Time] = []

        self.window = PygameWindow("Zed2i", size=self.SIZE)

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.subscriber = ros.Subscriber(
            "/zed2i", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )

    def __sub_callback(self, msg):
        pod: ZEDPOD = pickle.loads(msg.data)
        self.depth_image = pod.depth_image
        self.depth_map = pod.depth_map
        self.point_cloud = pod.point_cloud
        self.rgb_image = pod.rgb_image
        self.intrinsics = pod.intrinsics_matrix
        self.timestamps.append(pod.timestamp)
        while pod.timestamp - genpy.Duration(secs=1) > self.timestamps[0]:
            self.timestamps.pop(0)

    def run(self):
        while not ros.is_shutdown():
            if self.rgb_image is not None:
                fps = len(self.timestamps)
                latency = ros.Time.now() - self.timestamps[-1]
                latency_ms = int(latency.to_sec() * 1000)

                frame = (self.rgb_image * 255).astype(np.uint8)
                frame = cv2.resize(frame, self.SIZE)

                detect_and_visualize_charuco_pose(
                    frame,
                    intrinsics=self.intrinsics,
                    aruco_dict=AIRO_DEFAULT_ARUCO_DICT,
                    charuco_board=AIRO_DEFAULT_CHARUCO_BOARD,
                )
                frame = Image.fromarray(frame)
                frame = frame.resize(self.SIZE)
                frame = add_info_to_image(
                    frame,
                    title="Zed2i (RGB)",
                    frame_rate=f"{fps} Hz",
                    latency=f"{latency_ms} ms",
                )

                self.window.imshow(frame)

            self.rate.sleep()


def main():
    node = ZEDStreamRGB()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
