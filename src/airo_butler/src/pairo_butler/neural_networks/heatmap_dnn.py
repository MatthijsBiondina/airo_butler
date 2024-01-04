import pickle
from typing import Optional
from airo_butler.msg import PODMessage
import numpy as np
from pairo_butler.utils.pods import ImagePOD
from pairo_butler.utils.tools import pyout
import rospy as ros


class HeatmapDNN:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 60

    def __init__(self, name: str = "heatmap_dnn", cuda: bool = "true") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.subscriber: Optional[ros.Subscriber] = None

        self.frame: Optional[np.ndarray] = None
        self.frame_timestamp: Optional[ros.Time] = None

    def start_ros(self) -> None:
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.subscriber = ros.Subscriber(
            "/color_frame", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def __sub_callback(self, msg: PODMessage):
        """Callback function for receiving POD

        Args:
            msg (PODMessage): plain old data containing image and timestamp
        """
        pod: ImagePOD = pickle.loads(msg.data)

        self.frame = pod.image
        self.frame_timestamp = pod.timestamp

    def run(self):
        while not ros.is_shutdown():
            if self.frame is None or self.frame_timestamp is None:
                pass
            else:
                pass
            self.rate.sleep()


def main():
    node = HeatmapDNN(cuda=ros.get_param("-use_cuda", "false"))
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
