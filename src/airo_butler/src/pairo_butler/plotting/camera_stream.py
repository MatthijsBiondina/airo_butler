import pickle
from typing import List, Optional
import PIL
import cv2
import numpy as np
import genpy
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.pods import ImagePOD
from pairo_butler.plotting.plotting_utils import add_info_to_image


class CameraStream:
    QUEUE_SIZE: int = 10
    PUBLISH_RATE: int = 10

    def __init__(self, name: str = "camera_stream"):
        """Just streams the camera

        Args:
            name (str, optional): Name of the ros node. Defaults to "camera_stream".
        """
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.subscriber: Optional[ros.Subscriber] = None

        # Placeholder
        self.frame: Optional[PIL.Image] = None
        self.frame_timestamp: Optional[ros.Time] = None
        self.timestamps: List[ros.Time] = []

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
        self.timestamps.append(pod.timestamp)
        while pod.timestamp - genpy.Duration(secs=1) > self.timestamps[0]:
            self.timestamps.pop(0)

    def run(self):
        while not ros.is_shutdown():
            if self.frame is not None:
                fps: int = len(self.timestamps)
                latency: genpy.Duration = ros.Time.now() - self.frame_timestamp
                latency_ms = int(latency.to_sec() * 1000)

                image = add_info_to_image(
                    self.frame,
                    title="Camera Feed",
                    frame_rate=f"{fps} Hz",
                    latency=f"{latency_ms} ms",
                )

                cv2.imshow("/color_frame", np.array(image)[..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            self.rate.sleep()


def main():
    node = CameraStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
