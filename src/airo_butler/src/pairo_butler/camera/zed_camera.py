import os
import pickle
import sys
import time
from typing import List, Optional

import numpy as np
from airo_butler.msg import PODMessage
from pairo_butler.camera.towel_extremes_finder import TowelExtremesFinder
from pairo_butler.utils.pods import ZEDPOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i

CONFIDENCE_THRESHOLD = 100.0  # 0. -> no points, 100. -> all points


class ZEDClient:
    """
    A ROS client class for interfacing with a ZED camera.

    Attributes:
        QUEUE_SIZE (int): The size of the subscriber's queue.
        RATE (int): The rate at which the client operates.
        timeout (ros.Duration): The duration to wait for a message from the ZED camera.

    Methods:
        __init__(timeout=5): Initializes the ZED client with a specified timeout.
        pod: Property that returns the latest ZEDPOD received from the ZED camera.
        __sub_callback(msg): Callback function for the ROS subscriber.

    The ZEDClient class is responsible for subscribing to a ZED camera's data topic,
    handling the callback, and providing an interface to access the latest data. It
    includes functionality to handle timeouts and retrieve camera data.
    """

    QUEUE_SIZE = 2
    RATE = 30

    def __init__(self, timeout: int = 5):
        """
        Initializes the ZED client.

        This constructor sets up the ROS subscriber for the ZED camera's topic and
        initializes the timeout for receiving data.

        Args:
            timeout (int, optional): The maximum time in seconds to wait for a message
            from the ZED camera. Defaults to 5 seconds.
        """
        # Set the timeout duration for receiving messages from the ZED camera
        self.timeout = ros.Duration(timeout)
        # Initialize a ROS subscriber for the ZED camera's data topic
        self.subscriber: ros.Subscriber = ros.Subscriber(
            "/zed2i_topic", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        # Placeholder for the latest ZED camera data (ZEDPOD)
        self.__zed_pod: Optional[ZEDPOD] = None
        self.__timestamps: List[ros.Time] = []

    @property
    def pod(self) -> ZEDPOD:
        """
        Retrieves the latest ZEDPOD received from the ZED camera.

        This property waits until a ZEDPOD is received or until the timeout is reached.
        If no ZEDPOD is received within the timeout period, it logs an error, signals
        a ROS shutdown, and exits the program.

        Returns:
            ZEDPOD: The latest ZEDPOD object received from the ZED camera.
        """
        # Record the current time
        t0 = ros.Time.now()
        # Wait for ZEDPOD data until the timeout is reached
        while self.__zed_pod is None and ros.Time.now() < t0 + self.timeout:
            ros.sleep(1 / self.RATE)  # Sleep to allow other ROS callbacks to process
        # If no data is received, log an error, shutdown ROS, and exit the program
        # while self.__timestamps[-1] < t0 and ros.Time.now() < t0 + self.timeout:
        #     ros.sleep(1 / self.RATE)

        if self.__zed_pod is None:
            ros.logerr("No POD received from ZED. Is it running?")
            ros.signal_shutdown("Did not receive POD from ZED.")
            sys.exit(0)
        # Return the received ZEDPOD data
        return self.__zed_pod

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

    def __sub_callback(self, msg: PODMessage):
        """
        Callback function for the ROS subscriber.

        This function is automatically called when a new message is received on the
        subscribed topic. It deserializes the message and updates the internal ZEDPOD
        object.

        Args:
            msg (PODMessage): The message received from the topic.
        """
        # Deserialize the received message and update the ZEDPOD data
        pod = pickle.loads(msg.data)
        self.__zed_pod = pod
        self.__timestamps.append(pod.timestamp)
        while pod.timestamp - ros.Duration(secs=1) > self.__timestamps[0]:
            self.__timestamps.pop(0)


class ZED:
    """
    A ROS node class for interfacing with a ZED 2i camera.

    Attributes:
        PUBLISH_RATE (int): The rate at which the node publishes messages.
        QUEUE_SIZE (int): The size of the message queue for the publishers.
        node_name (str): The name of the ROS node.
        rate (ros.Rate): ROS rate object for controlling the publish frequency.
        zed (Zed2i): Instance of the Zed2i camera object.
        rgb_publisher (ros.Publisher): ROS publisher for RGB images.
        cloud_publisher (ros.Publisher): ROS publisher for point cloud data.
        publisher (ros.Publisher): General ROS publisher for the node.

    Methods:
        __init__(name="Zed2i"): Constructor to initialize the ZED camera and ROS node.
        start_ros(): Initializes ROS node and publishers.
        run(): Main execution loop for capturing and publishing camera data.
    """

    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "Zed2i") -> None:
        """
        Initializes the ZED ROS node.

        This constructor sets up the ZED 2i camera with specified configurations and
        initializes necessary variables for the ROS node.

        Args:
            name (str, optional): The name of the ROS node. Defaults to "Zed2i".
        """
        # Setting the name for the ROS node
        self.node_name: str = name
        # Initializing ROS rate to None, to be set in start_ros
        self.rate: Optional[ros.Rate] = None
        # Initialize the ZED 2i camera with specified resolution, depth mode, and FPS
        # self.zed = Zed2i(
        #     Zed2i.RESOLUTION_1080, depth_mode=Zed2i.QUALITY_DEPTH_MODE, fps=10
        # )
        self.zed = Zed2i(
            Zed2i.RESOLUTION_1080, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=10
        )
        # Placeholder for ROS publishers
        self.rgb_publisher: Optional[ros.Publisher] = None
        self.cloud_publisher: Optional[ros.Publisher] = None
        self.publisher: Optional[ros.Publisher] = None

        self.towel_extremes_finder = TowelExtremesFinder()

    def start_ros(self):
        """
        Initializes the ROS node and publishers for the ZED camera.

        This method sets up the ROS node with the specified name and logging level,
        initializes ROS publishers for different types of data (RGB images, point clouds,
        etc.), and logs a confirmation message once the node is successfully started.
        """
        # Initialize the ROS node with the specified node name and log level
        ros.init_node(self.node_name, log_level=ros.INFO)
        # Set the publish rate for the node
        self.rate = ros.Rate(self.PUBLISH_RATE)
        # Initialize the ROS publisher with the specified topic, message type, and
        # queue size
        self.publisher = ros.Publisher(
            "/zed2i_topic", PODMessage, queue_size=self.QUEUE_SIZE
        )
        # Log an information message indicating successful initialization
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        """
        The main execution loop of the ROS node for the ZED camera.

        This method continuously captures data from the ZED 2i camera, including RGB
        images and point clouds, and publishes them to the designated ROS topics. The
        loop runs until ROS is shutdown, and it uses a rate limiter to control the
        frequency of data publishing.
        """
        # Main loop running until ROS is shutdown
        while not ros.is_shutdown():
            try:
                # Capture data from ZED camera and form a PODMessage

                point_cloud = self.__preprocess_point_cloud()[:, :3]
                self.towel_extremes_finder.process_point_cloud(point_cloud)

                msg = ZEDPOD(
                    rgb_image=self.zed.get_rgb_image(),
                    # point_cloud=self.__preprocess_point_cloud(),
                    intrinsics_matrix=self.zed.intrinsics_matrix(),
                    timestamp=ros.Time.now(),
                )
                # Publish the formed PODMessage
                publish_pod(self.publisher, msg)
                # Sleep to maintain the publish rate
            except IndexError as e:
                ros.logwarn(f"{e}")
                self.zed = Zed2i(
                    Zed2i.RESOLUTION_1080, depth_mode=Zed2i.QUALITY_DEPTH_MODE, fps=10
                )
            self.rate.sleep()

    def __preprocess_point_cloud(self):
        point_cloud = self.zed.get_colored_point_cloud()
        confidence_map = self.zed._retrieve_confidence_map()

        points = point_cloud.points[confidence_map.reshape(-1) < CONFIDENCE_THRESHOLD]
        colors = point_cloud.colors[confidence_map.reshape(-1) < CONFIDENCE_THRESHOLD]

        return np.concatenate((points, colors), axis=1)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    node = ZED()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
