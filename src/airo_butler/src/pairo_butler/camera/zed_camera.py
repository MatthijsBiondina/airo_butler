from typing import Optional
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import ZEDPOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter


class ZED:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "zed_camera") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        self.zed = Zed2i(
            Zed2i.RESOLUTION_720, depth_mode=Zed2i.QUALITY_DEPTH_MODE, fps=30
        )

        self.publisher: Optional[ros.Publisher] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.publisher = ros.Publisher("/zed", PODMessage, queue_size=self.QUEUE_SIZE)

    def run(self):
        while not ros.is_shutdown():
            pod = ZEDPOD(
                self.zed.get_rgb_image(),
                self.zed.get_colored_point_cloud(),
                self.zed.get_depth_image(),
                self.zed.get_depth_map(),
                ros.Time.now(),
            )
            publish_pod(self.publisher, pod)
            self.rate.sleep()


def main():
    node = ZED()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()