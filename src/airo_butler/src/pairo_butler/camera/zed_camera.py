import time
from typing import Optional
from airo_butler.msg import PODMessage, NPMessage
from pairo_butler.utils.np_messages import publish_np_array
from pairo_butler.utils.pods import ZEDPOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i


class ZED:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "Zed2i") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # self.zed = Zed2i(
        #     Zed2i.RESOLUTION_1080, depth_mode=Zed2i.QUALITY_DEPTH_MODE, fps=30
        # )
        self.zed = Zed2i(
            Zed2i.RESOLUTION_1080, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=30
        )

        self.rgb_publisher: Optional[ros.Publisher] = None
        self.cloud_publisher: Optional[ros.Publisher] = None
        self.publisher: Optional[ros.Publisher] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.publisher = ros.Publisher("/zed2i", PODMessage, queue_size=self.QUEUE_SIZE)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            msg = ZEDPOD(
                rgb_image=self.zed.get_rgb_image(),
                point_cloud=self.zed.get_colored_point_cloud(),
                intrinsics_matrix=self.zed.intrinsics_matrix(),
                timestamp=ros.Time.now(),
            )
            publish_pod(self.publisher, msg)
            self.rate.sleep()


def main():
    node = ZED()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
