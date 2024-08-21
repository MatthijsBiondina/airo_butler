import sys
from pairo_butler.utils.tools import load_config, pyout
import pickle
import sys
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
from pairo_butler.utils.pods import ImagePOD, publish_pod

from airo_butler.srv import Reset
import rospy as ros
import pyrealsense2
from airo_butler.msg import PODMessage
import sys


class RS2Client:
    """
    A ROS client class for interfacing with a RealSense2 (RS2) camera.

    Attributes:
        QUEUE_SIZE (int): The size of the subscriber's queue.
        RATE (int): The rate at which the client operates.
        timeout (int): The maximum time to wait for the RS2 service to be available.

    Methods:
        __init__(timeout=5): Initializes the RS2 client with a specified timeout.
        pod: Property that returns the latest ImagePOD received from the RS2 camera.
        __sub_callback(msg): Callback function for the ROS subscriber.
        __signal_rs2_reset(): Signals a hardware reset to the RS2 camera.

    The RS2Client class is responsible for subscribing to a RealSense2 camera's color frame topic, handling the callback, and providing an interface to access the latest image data. It includes functionality to reset the RS2 camera and handle timeouts if the camera or service is unavailable.
    """

    QUEUE_SIZE = 2
    RATE = 30

    def __init__(self, timeout: int = 5):
        """
        Initializes the RS2 client.

        This constructor sets up the RS2 client, signals a reset to the RS2 camera, and
        initializes a ROS subscriber to the '/color_frame' topic.

        Args:
            timeout (int): The maximum time in seconds to wait for the RS2 service to be
            available. Default is 5 seconds.
        """

        try:
            self.timeout = ros.Duration(timeout)
        except TypeError:
            self.timeout = False

        self.__signal_rs2_reset()

        self.subscriber: ros.Subscriber = ros.Subscriber(
            "/rs2_topic", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )

        # Placeholders
        self.config = None
        self.__rs2_pod: Optional[ImagePOD] = None
        self.__timestamps: List[ros.Time] = []

    @property
    def pod(self) -> ImagePOD:
        """
        Retrieves the latest ImagePOD received from the RS2 camera.

        This property waits until an ImagePOD is received or until the timeout is reached.
        If no ImagePOD is received within the timeout period, it logs an error, signals a
        ROS shutdown, and exits.

        Returns:
            The latest ImagePOD object received from the RS2 camera.
        """
        t0 = ros.Time.now()
        while self.__rs2_pod is None and (
            not self.timeout or ros.Time.now() < t0 + self.timeout
        ):
            ros.sleep(1 / self.RATE)
        if self.__rs2_pod is None:
            ros.logerr("Did not recieve pod from RS2. Is it running?")
            ros.signal_shutdown("Did not receive pod from RS2.")
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

    def reset(self):
        self.__signal_rs2_reset()

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

    def __signal_rs2_reset(self):
        """
        Signals a hardware reset to the RS2 camera.

        This function waits for the 'reset_realsense_service' to become available within
        the specified timeout. If the service is not available, it logs an error, signals
        a ROS shutdown, and exits. If the service is available, it calls the service to
        reset the camera.
        """

        try:
            # Check whether the service is running
            ros.wait_for_service("reset_realsense_service", timeout=self.timeout)
        except ros.exceptions.ROSException:
            # If not, exit with error message
            ros.logerr(f"Cannot connect to RS2 server. Is it running?")
            ros.signal_shutdown("Cannot connect to RS2 server.")
            sys.exit(0)
        except ValueError:
            return

        # Connect to and call reset service.
        service = ros.ServiceProxy("reset_realsense_service", Reset)
        resp = service()

        # If reset unsuccessful, exit with error message.
        if not resp.success:
            ros.logerr(f"Failed to reset rs2 camera.")
            ros.signal_shutdown("Failed to reset rs2 camera.")
            sys.exit(0)


class RS2_camera:
    # Tuple of tuples containing possible resolutions for the RealSense2 camera
    RESOLUTIONS = ((640, 480), (1280, 720))
    # Queue size for ROS publisher
    QUEUE_SIZE = 2
    RATE = 5

    def __init__(
        self,
        name: str = "rs2",
        fps: int = 15,
        rs2_resolution: Tuple[int, int] = (640, 480),
        out_resolution: int = 720,
        serial_number: str = "925322060348",
    ):
        """
        Initializes a new instance of the ROS node for interfacing with a RealSense2
        camera. This constructor sets up the necessary parameters for the RealSense2 camera,
        including its resolution and frame rate. It also initializes ROS-related properties
        such as the node name, publisher, and rate.

        Args:
            name (str, optional): The name of the ROS node. This name is used for identifying the node within the ROS ecosystem. Defaults to "rs2".
            fps (int, optional): The frame per second rate at which the camera captures and publishes images. It defines the frequency of the camera's data stream. Defaults to 30.
            rs2_resolution (Tuple[int, int], optional): The resolution setting of the RealSense2 camera. It is a tuple indicating the width and height of the camera frame in pixels. Defaults to (640, 480).
            out_resolution (int, optional): The resolution of the processed square image output. This parameter defines the dimensions to which the camera images will be resized. Defaults to 512.

        Raises:
            AssertionError: If the specified `rs2_resolution` is not within the predefined list of allowed resolutions.
        """
        self.rs2_resolution = rs2_resolution
        self.fps = fps
        self.out_resolution = out_resolution
        # ROS HOUSEKEEPING
        # Set the node name as provided or default.
        self.node_name = sys.argv[1].split(":=")[1] if len(sys.argv) > 1 else name

        self.rate: ros.Rate
        self.publisher: ros.Publisher

        # Check if the resolution is valid as per predefined list.
        assert rs2_resolution in self.RESOLUTIONS
        # Create a pipeline for RealSense2 camera data.
        self.pipeline = pyrealsense2.pipeline()

        self.serial_number = serial_number

        # Placeholders
        self.__last_reset: Optional[ros.Time] = None
        self.__reset: bool = False
        self.__reset_result: Optional[bool] = None

    def start_ros(self):
        """
        Initializes the ROS node and sets up the publisher for the RealSense2 camera
        frames. This method is responsible for initializing the ROS node with the
        specified name and logging level. It sets up the ROS publisher with the defined
        topic name, message type, and queue size. It also initializes the ROS rate object
        used to control the rate of frame publishing. The method logs an information
        message once the ROS node initialization is successful.
        """
        # Initialize the ROS node with the specified node name and log level.
        ros.init_node(self.node_name, log_level=ros.INFO)

        serial_number = ros.get_param("~serial_number").replace("SN", "")
        context = pyrealsense2.context()
        device_str = "CONNECTED DEVICES:"
        for device in context.devices:
            device_sn = device.get_info(pyrealsense2.camera_info.serial_number)
            device_str += (
                f"\n{'[X]' if device_sn == serial_number else '[ ]'}  {device}"
            )

        pyout(device_str)
        self.fps = int(ros.get_param("fps", self.fps))
        if ros.get_param("~hd", False):
            self.rs2_resolution = self.RESOLUTIONS[-1]

        # Configure the pipeline to stream color data with given resolution and frame rate.
        config = pyrealsense2.config()
        config.enable_device(str(serial_number))
        # self.profile = config.resolve(self.pipeline)

        config.enable_stream(
            pyrealsense2.stream.color,
            *self.rs2_resolution,
            pyrealsense2.format.rgb8,
            self.fps,
        )
        config.enable_stream(
            pyrealsense2.stream.depth,
            *self.rs2_resolution,
            pyrealsense2.format.z16,
            self.fps,
        )
        self.rs2config = config

        # Start the camera pipeline.
        self.pipeline.start(config)
        # Set the rate at which frames will be published.
        self.publish_rate = self.fps
        # Set the output image resolution.
        self.resolution = self.out_resolution

        self.intrinsics_matrix = self.__compute_intrinsics_matrix()

        # Set reset time
        self.__last_reset = ros.Time.now()
        # Set the publish rate for the node.
        self.rate = ros.Rate(self.publish_rate)
        # Initialize the ROS publisher with the specified topic, message type, and queue size.
        self.pub_name = f'/{ros.get_param("topic", "rs2_topic")}'

        self.publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        # Initialize reset procedure
        self.reset_service = ros.Service(
            "reset_realsense_service", Reset, self.toggle_reset
        )

        self.config = load_config()

        # Log an information message indicating successful initialization.
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        """
        The main execution loop of the ROS node for capturing and publishing RealSense2 camera
        frames. This method continuously captures color frames from the RealSense2 camera,
        converts them to PIL images, and encapsulates them in ImagePOD objects with a
        timestamp. These ImagePODs are then published to the designated ROS topic. The loop
        runs until ROS is shutdown. The method uses a rate limiter to control the
        frequency of frame publishing.
        """
        # Continuously run until ROS is shutdown.
        while not ros.is_shutdown():
            if self.__reset:
                self.reset_camera()

            # Wait and capture a color frame from the RealSense2 camera.
            try:
                frameset = self.pipeline.wait_for_frames()
            except RuntimeError:
                self.reset_camera()
                continue

            # Convert the captured frame to a PIL image.
            image = self.__frame2pillow(frameset.get_color_frame())

            align = pyrealsense2.align(pyrealsense2.stream.color)
            aligned_frames = align.process(frameset)
            color_image = self.__frame2pillow(aligned_frames.get_color_frame())
            depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())

            # depth_frame = (
            #     1 - np.clip(depth_frame / self.config.max_distance, 0, 1)
            # ) * 255
            # depth_frame[depth_frame == 255] = 0
            # depth_image = self.__frame2pillow(depth_frame)

            # Create an ImagePOD object with the image and current timestamp.
            pod = ImagePOD(
                color_frame=color_image,
                depth_frame=depth_frame,
                image=image,
                intrinsics_matrix=self.intrinsics_matrix,
                timestamp=ros.Time.now(),
            )
            # Publish the ImagePOD object to the designated ROS topic.
            publish_pod(self.publisher, pod)
            # Sleep for a while as per the set rate to control the publishing frequency.
            self.rate.sleep()

    def toggle_reset(self, msg):
        """
        Toggles the reset state for the RealSense2 Camera and waits for the reset process
        to complete.

        This function sets the reset flag to True, initiating the reset process in another
        function. It then waits until the reset result is available, checking at the
        frequency of `publish_rate`.

        Args:
            msg: The message received (unused in this function but required for ROS
            service structure).

        Returns:
            The result of the reset process (True if successful, False otherwise).
        """
        # Initialize reset result to None and flag reset as True to start the process
        self.__reset_result = None
        self.__reset = True

        # Wait in a loop until the reset result is updated
        while self.__reset_result is None:
            ros.sleep(
                1 / self.publish_rate
            )  # Sleep to prevent blocking at the frequency of publish_rate

        # Return the result of the reset process
        return self.__reset_result

    def reset_camera(self, forced: bool = False):
        """
        Performs the actual hardware reset of the RealSense2 Camera.

        This function stops the camera pipeline, restarts it, and waits for a brief
        moment to ensure the reset is complete. It updates the reset result flag based
        on the success of these operations. If an exception occurs during the reset, it
        logs an error and sets the reset result to False.

        Exceptions are caught and logged, and the reset result is set accordingly.
        """
        try:
            if not forced and (ros.Time.now() < self.__last_reset + ros.Duration(10)):
                self.__reset = False
                self.__reset_result = True
            else:
                # Log the initiation of the reset process
                ros.loginfo("Resetting RealSense2 Camera...")

                # Stop and restart the camera pipeline to perform the reset
                self.pipeline.stop()
                ros.sleep(1)
                self.pipeline.start(self.rs2config)

                # Wait for a brief moment after restarting the pipeline
                ros.sleep(1)

                # Reset process is complete, update flags
                self.__reset = False
                self.__reset_result = True
                self.__last_reset = ros.Time.now()

        except Exception as e:
            # Log any exception during reset and update the reset result to False
            ros.logerr(f"Failed to reset rs2: {e}")
            self.__reset = False
            self.__reset_result = False

    def __frame2pillow(self, frame: pyrealsense2.frame | np.ndarray):
        """
        Converts a frame captured from the RealSense2 camera to a PIL (Python Imaging
        Library) image.

        This private method transforms a RealSense2 frame into a numpy array, then converts
        this array to a PIL image for further processing. The image is first cropped to a
        square format based on the smaller dimension (height or width) and then resized to
        the specified resolution.

        Args:
            frame (pyrealsense2.frame): A frame captured from the RealSense2 camera.

        Returns:
            PIL.Image: The processed image in PIL format, cropped and resized to the specified resolution.
        """
        # Convert the RealSense2 frame to a numpy array.
        if isinstance(frame, np.ndarray):
            img_array = frame.astype(np.uint8)
        else:
            img_array = np.asanyarray(frame.get_data()).astype(np.uint8)
        # Create a PIL image from the numpy array.
        image = Image.fromarray(img_array)

        # Calculate the dimensions for cropping the image to a square.
        crop = min(image.width, image.height)
        left = round((image.width - crop) / 2)
        top = round((image.height - crop) / 2)
        right = round((image.width + crop) / 2)
        bottom = round((image.height + crop) / 2)
        # Crop the image to the calculated dimensions.
        image = image.crop((left, top, right, bottom))

        # Resize the cropped image to the desired resolution.
        image = image.resize((self.resolution, self.resolution))

        return image  # Return the processed PIL image.

    def __compute_intrinsics_matrix(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Get the intrinsics of the color stream
        color_profile = color_frame.get_profile()
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        original_fx = intrinsics.fx
        original_fy = intrinsics.fy
        original_cx = intrinsics.ppx
        original_cy = intrinsics.ppy
        original_width = intrinsics.width  # Original width before crop
        original_height = intrinsics.height  # Original height before crop

        # Assuming self.resolution is the new size after cropping and resizing
        new_width = self.resolution
        new_height = self.resolution

        # Calculate crop dimensions
        crop = min(original_width, original_height)
        left = round((original_width - crop) / 2)
        top = round((original_height - crop) / 2)

        # Adjust principal point for the crop
        cx_prime = original_cx - left
        cy_prime = original_cy - top

        # Calculate scale factors
        scale_factor_x = new_width / crop
        scale_factor_y = new_height / crop

        # Adjust focal lengths and principal points for resize
        fx_prime = original_fx * scale_factor_x
        fy_prime = original_fy * scale_factor_y
        cx_prime *= scale_factor_x
        cy_prime *= scale_factor_y

        # New camera matrix
        new_camera_matrix = np.array(
            [[fx_prime, 0, cx_prime], [0, fy_prime, cy_prime], [0, 0, 1]]
        )

        return new_camera_matrix


def main():
    node = RS2_camera()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
