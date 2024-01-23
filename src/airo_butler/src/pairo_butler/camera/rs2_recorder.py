import pickle
import sys
from typing import List, Optional

import numpy as np
from pairo_butler.utils.pods import ImagePOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
import pyrealsense2
from airo_butler.msg import PODMessage
from airo_butler.srv import Reset
from PIL import Image


class RecorderClient:
    QUEUE_SIZE = 2
    RATE = 30

    def __init__(self, timeout: int = 5):
        self.timeout = ros.Duration(timeout)

        self.signal_reset()
        self.subscriber: ros.Subscriber = ros.Subscriber(
            "/recorder_topic",
            PODMessage,
            self.__sub_callback,
            queue_size=self.QUEUE_SIZE,
        )

        # Placeholders
        self.__rs2_pod: Optional[ImagePOD] = None
        self.__timestamps: List[ros.Time] = []

    @property
    def pod(self) -> ImagePOD:
        t0 = ros.Time.now()
        while self.__rs2_pod is None and ros.Time.now() < t0 + self.timeout:
            ros.sleep(1 / self.RATE)
        if self.__rs2_pod is None:
            ros.logerr("Did not recieve pod from recorder. Is it running?")
            ros.signal_shutdown("Did not receive pod from recorder.")
            sys.exit(0)
        return self.__rs2_pod

    @property
    def fps(self) -> int:
        """
        Computes the frame-rate at which frames are recieved.

        Returns:
            int: Frames per second of the RealSense2
        """
        return len(self.__timestamps)

    @property
    def latency(self) -> int:
        """
        Computes the latency of incoming frames.

        Returns:
            int: latency in milliseconds
        """
        try:
            latency = ros.Time.now() - self.__timestamps[-1]
            return int(latency.to_sec() * 1000)
        except IndexError:
            return -1  # -1 indicates no frames recieved in the last second.

    def signal_reset(self):
        RS2_Recorder.signal_reset()

    def __sub_callback(self, msg: PODMessage):
        """
        Callback function for the ROS subscriber.

        This function is automatically called when a new message is received on the
        '/rs2_topic' topic. It deserializes the message and updates the internal
        ImagePOD object.

        Args:
            msg: The message received from the topic.
        """
        pod = pickle.loads(msg.data)

        self.__rs2_pod = pod
        self.__timestamps.append(pod.timestamp)
        while pod.timestamp - ros.Duration(secs=1) > self.__timestamps[0]:
            self.__timestamps.pop(0)


class RS2_Recorder:
    RESOLUTION = (640, 480)
    QUEUE_SIZE = 2

    def __init__(
        self,
        name: str = "rs2_recording",
        fps: int = 15,
        serial_number: str = "944122073290",
    ):
        self.serial_number = serial_number
        self.fps = fps
        self.node_name: str = name
        self.pub_name: str = "/recorder_topic"
        self.rate: Optional[ros.Rate] = None
        self.publisher: Optional[ros.Publisher] = None

        self.pipeline = pyrealsense2.pipeline()
        config = pyrealsense2.config()
        config.enable_device(serial_number)
        config.enable_stream(
            pyrealsense2.stream.color, *self.RESOLUTION, pyrealsense2.format.rgb8, fps
        )
        self.pipeline.start(config)
        self.publish_rate = fps

        # Placeholders
        self.__last_reset: Optional[ros.Time] = None
        self.__reset: bool = False
        self.__reset_result: Optional[bool] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.__last_reset = ros.Time.now()
        self.rate = ros.Rate(self.publish_rate)
        self.publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.reset_servcie = ros.Service(
            "reset_recorder_service", Reset, self.toggle_reset
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.__reset:
                self.reset_camera()

            frame = self.pipeline.wait_for_frames().get_color_frame()
            image = Image.fromarray(np.asanyarray(frame.get_data()).astype(np.uint8))

            pod = ImagePOD(
                image=image, intrinsics_matrix=None, timestamp=ros.Time.now()
            )

            publish_pod(self.publisher, pod)
            self.rate.sleep()

    def toggle_reset(self, msg):
        self.__reset_result = None
        self.__reset = True

        while self.__reset_result is None:
            ros.sleep(1 / self.publish_rate)

        return self.__reset_result

    def reset_camera(self):
        try:
            if ros.Time.now() < self.__last_reset + ros.Duration(10):
                self.__reset = False
                self.__reset_result = True
            else:
                self.pipeline.stop()
                ros.sleep(1)
                self.pipeline.start()
                ros.sleep(1)
                self.__reset = False
                self.__reset_result = True
                self.__last_reset = ros.Time.now()

        except Exception as e:
            ros.logerr(f"Failed to reset rs2 ({self.serial_number}): {e}")
            self.__reset = False
            self.__reset_result = False

    @staticmethod
    def signal_reset():
        try:
            ros.wait_for_service("reset_recorder_service", timeout=5)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to Recorder server. Is it running?")
            ros.signal_shutdown("Cannot connect to Recorder server.")
            sys.exit(0)

        service = ros.ServiceProxy("reset_recorder_service", Reset)
        resp = service()

        if not resp.success:
            ros.logerr(f"Failed to reset recorder.")
            ros.signal_shutdown("Failed to reset recorder")
            sys.exit(0)


def main():
    node = RS2_Recorder()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
