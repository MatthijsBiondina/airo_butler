import pygame
import pickle
from typing import List, Optional
import PIL
import cv2
import numpy as np
from pairo_butler.plotting.pygame_plotter import PygameWindow
import genpy
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import ImagePOD
from pairo_butler.plotting.plotting_utils import add_info_to_image


class CameraStream:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 30

    def __init__(self, name: str = "camera_stream"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.subscriber: Optional[ros.Subscriber] = None
        self.frame: Optional[PIL.Image] = None
        self.frame_timestamp: Optional[ros.Time] = None
        self.timestamps: List[ros.Time] = []

        # Pygame initialization
        self.window = PygameWindow("Realsense2 (RGB)", (512, 512))

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.subscriber = ros.Subscriber(
            "/color_frame", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def __sub_callback(self, msg: PODMessage):
        pod: ImagePOD = pickle.loads(msg.data)
        self.frame = pod.image
        self.frame_timestamp = pod.timestamp
        self.timestamps.append(pod.timestamp)
        while pod.timestamp - genpy.Duration(secs=1) > self.timestamps[0]:
            self.timestamps.pop(0)

    def run(self):
        while not ros.is_shutdown():
            if self.frame is not None:
                fps = len(self.timestamps)
                latency = ros.Time.now() - self.frame_timestamp
                latency_ms = int(latency.to_sec() * 1000)

                frame = add_info_to_image(
                    self.frame,
                    title="RealSense2 (RGB)",
                    frame_rate=f"{fps} Hz",
                    latency=f"{latency_ms} ms",
                )
                self.window.imshow(frame)
            self.rate.sleep()


def main():
    node = CameraStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
