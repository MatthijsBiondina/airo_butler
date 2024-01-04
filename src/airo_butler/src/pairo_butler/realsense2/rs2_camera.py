from typing import Optional, Tuple
from PIL import Image
import numpy as np
from pairo_butler.utils.pods import ImagePOD, publish_pod
from pairo_butler.utils.tools import pyout
import rospy as ros
import pyrealsense2
from airo_butler.msg import PODMessage


class RS2_camera:
    RESOLUTIONS = ((640, 480), (960, 540), (1280, 720))
    QUEUE_SIZE = 2

    def __init__(
            self,
            name: str = "rs2_camera",
            fps: int = 30,
            rs2_resolution: Tuple[int, int] = (640, 480),
            out_resolution: int = 512,
    ):
        # ROS housekeeping
        self.node_name: str = name
        self.pub_name: str = "/rs2_image"
        self.rate: Optional[ros.Rate] = None
        self.publisher: Optional[ros.Publisher] = None

        # start realsense2 camera
        assert rs2_resolution in self.RESOLUTIONS
        self.pipeline = pyrealsense2.pipeline()
        pyrealsense2.config().enable_stream(
            pyrealsense2.stream.color, *rs2_resolution,
            pyrealsense2.format.bgr8, fps
        )
        self.pipeline.start()
        self.publish_rate = fps
        self.resolution = out_resolution

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.publish_rate)
        self.publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo("RS2_camera: OK!")

    def run(self):
        while not ros.is_shutdown():
            frame = self.pipeline.wait_for_frames().get_color_frame()
            image = self.__frame2pillow(frame)
            pod = ImagePOD(image=image, timestamp=ros.Time.now())
            publish_pod(self.publisher, pod)
            self.rate.sleep()

    def __frame2pillow(self, frame: pyrealsense2.frame):
        img_array = np.asanyarray(frame.get_data()).astype(np.uint8)
        image = Image.fromarray(img_array)

        # Calculate crop box and crop image to square
        crop = min(image.width, image.height)
        left = round((image.width - crop) / 2)
        top = round((image.height - crop) / 2)
        right = round((image.width + crop) / 2)
        bottom = round((image.height + crop) / 2)
        image = image.crop((left, top, right, bottom))

        # Resize image
        image = image.resize((self.resolution, self.resolution))

        return image


def main():
    node = RS2_camera()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
