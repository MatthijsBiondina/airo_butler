from typing import Optional, Tuple
from PIL import Image
import numpy as np
from pairo_butler.utils.pods import ImagePOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
import pyrealsense2
from airo_butler.msg import PODMessage


class RS2_camera:
    # Tuple of tuples containing possible resolutions for the RealSense2 camera
    RESOLUTIONS = ((640, 480), (960, 540), (1280, 720))
    # Queue size for ROS publisher
    QUEUE_SIZE = 2

    def __init__(
        self,
        name: str = "rs2",
        fps: int = 30,
        rs2_resolution: Tuple[int, int] = (640, 480),
        out_resolution: int = 512,
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

        # ROS HOUSEKEEPING
        # Set the node name as provided or default.
        self.node_name: str = name
        # Define the publication topic for camera frames.
        self.pub_name: str = "/color_frame"
        # Initialize a ROS rate object, to be set in 'start_ros' method.
        self.rate: Optional[ros.Rate] = None
        # Initialize a ROS publisher, to be set in 'start_ros' method.
        self.publisher: Optional[ros.Publisher] = None

        # START REALSENSE2 CAMERA
        # Check if the resolution is valid as per predefined list.
        assert rs2_resolution in self.RESOLUTIONS
        # Create a pipeline for RealSense2 camera data.
        self.pipeline = pyrealsense2.pipeline()
        # Configure the pipeline to stream color data with given resolution and frame rate.
        pyrealsense2.config().enable_stream(
            pyrealsense2.stream.color, *rs2_resolution, pyrealsense2.format.bgr8, fps
        )
        # Start the camera pipeline.
        self.pipeline.start()
        # Set the rate at which frames will be published.
        self.publish_rate = fps
        # Set the output image resolution.
        self.resolution = out_resolution

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
        # Set the publish rate for the node.
        self.rate = ros.Rate(self.publish_rate)
        # Initialize the ROS publisher with the specified topic, message type, and queue size.
        self.publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        # Log an information message indicating successful initialization.
        ros.loginfo("RS2_camera: OK!")

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
            # Wait and capture a color frame from the RealSense2 camera.
            frame = self.pipeline.wait_for_frames().get_color_frame()
            # Convert the captured frame to a PIL image.
            image = self.__frame2pillow(frame)
            # Create an ImagePOD object with the image and current timestamp.
            pod = ImagePOD(image=image, timestamp=ros.Time.now())
            # Publish the ImagePOD object to the designated ROS topic.
            publish_pod(self.publisher, pod)
            # Sleep for a while as per the set rate to control the publishing frequency.
            self.rate.sleep()

    def __frame2pillow(self, frame: pyrealsense2.frame):
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


def main():
    node = RS2_camera()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
