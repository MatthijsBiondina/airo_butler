import pickle
from typing import Optional

import cv2
import numpy as np

import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import ImagePOD


class CameraStream:
    QUEUE_SIZE: int = 10
    PUBLISH_RATE: int = 10

    def __init__(self, name: str = "camera_stream"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.subscriber: Optional[ros.Subscriber] = None

        # Placeholder
        self.img: Optional[np.ndarray] = None

    def start_ros(self):
        """
        Create ros node for this class and initialize subscribers and
        publishers
        """
        # Create a ROS node with a name for this class
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        # Init subscriber
        self.subscriber = ros.Subscriber(
            "/rs2_image", PODMessage, self.__callback,
            queue_size=self.QUEUE_SIZE
        )

    def __callback(self, msg: PODMessage):
        """
        Cleanly close the node
        """
        pod: ImagePOD = pickle.loads(msg.data)

        self.img = np.array(pod.image)[..., ::-1]

    def run(self):
        while not ros.is_shutdown():
            if self.img is not None:
                cv2.imshow("RealSense2", self.img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.rate.sleep()


def main():
    node = CameraStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
