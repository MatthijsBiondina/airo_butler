import time
from typing import List, Optional

from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from airo_butler.msg import PODMessage
import genpy
from pairo_butler.plotting.plotting_utils import add_info_to_image
from pairo_butler.utils.tools import pyout
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

        self.timestamps: List[ros.Time] = []

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.collector = ROSMessageCollector(exact=["/heatmap", "/color_frame"])
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            messages = self.collector.next(timeout=1)
            if messages:
                timestamp: ros.Time = messages["/color_frame"].timestamp
                image: Image = messages["/color_frame"].image
                heatmap: np.ndarray = messages["/heatmap"].array
                self.__update_timestamps(timestamp)
                overlay_image = self.__overlay_heatmap_on_image(image, heatmap)

                fps: int = len(self.timestamps)
                latency: genpy.duration = ros.Time.now() - timestamp
                latency_ms = int(latency.to_sec() * 1000)

                overlay_image = add_info_to_image(
                    overlay_image,
                    title="Heatmap",
                    frame_rate=f"{fps} Hz",
                    latency=f"{latency_ms} ms",
                )

                cv2.imshow("/heatmap", np.array(overlay_image)[..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            self.rate.sleep()

    def __update_timestamps(self, new_stamp):
        self.timestamps.append(new_stamp)
        while new_stamp - genpy.Duration(secs=1) > self.timestamps[0]:
            self.timestamps.pop(0)

    def __overlay_heatmap_on_image(self, image: Image, heatmap: np.ndarray):
        heatmap_normalized = (heatmap) / (np.sum(heatmap) + 1e-6)
        colormap = plt.get_cmap("viridis")
        heatmap_colored = colormap(heatmap_normalized)

        # Convert to PIL image and ensure same size as original
        heatmap_image = Image.fromarray(
            (heatmap_colored * 255).astype(np.uint8)
        ).convert("RGBA")
        heatmap_image.putalpha(127)

        # Overlay the heatmap on the image
        overlay_image = Image.new("RGBA", image.size)
        overlay_image = Image.alpha_composite(overlay_image, image.convert("RGBA"))
        overlay_image = Image.alpha_composite(overlay_image, heatmap_image)

        # Convert to rgb
        background = Image.new("RGB", overlay_image.size, (255, 255, 255))
        rgb_image = Image.alpha_composite(
            background.convert("RGBA"), overlay_image
        ).convert("RGB")

        return rgb_image


def main():
    node = HeatmapStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
