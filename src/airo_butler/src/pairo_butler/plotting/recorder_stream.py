from typing import Optional

import PIL
import numpy as np
from pairo_butler.plotting.plotting_utils import add_info_to_image
from pairo_butler.plotting.pygame_plotter import PygameWindow
import rospy as ros
from pairo_butler.camera.rs2_recorder import RecorderClient


class RecorderStream:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 30

    def __init__(self, name: str = "recorder_stream"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.rec: Optional[RecorderClient] = None

        self.window = PygameWindow("Recorder (RGB)", (640, 480))

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.rec = RecorderClient()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            frame = self.rec.pod.image
            frame = add_info_to_image(
                self.rec.pod.image,
                title="Recorder (RGB)",
                frame_rate=f"{self.rec.fps} Hz",
                latency=f"{self.rec.latency} ms",
            )
            self.window.imshow(frame)
            self.rate.sleep()


def main():
    node = RecorderStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
