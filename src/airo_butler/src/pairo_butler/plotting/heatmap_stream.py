from typing import Optional
from pairo_butler.utils.ros_message_collector import ROSMessageCollector
import rospy as ros


class HeatmapStream:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 30

    def __init__(self, name: str = "heatmap_stream"):
        """Streams keypoint heatmap overlayed on input frame

        Args:
            name (str, optional): Name of the ros node. Defaults to "heatmap_stream".
        """
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.collector: Optional[ROSMessageCollector] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.collector = ROSMessageCollector(exact=["/color_frame", "/heatmap"])
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            self.rate.sleep()


def main():
    node = HeatmapStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
