from typing import List
import numpy as np
from pairo_butler.plotting.plotting_utils import (
    add_info_to_image,
    compute_fps_and_latency,
    overlay_heatmap_on_image,
)
from pairo_butler.data.timesync import TimeSync
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout
from PIL import Image
import cv2


class KeypointHeatmapStream:
    RATE = 30

    def __init__(self, name: str = "keypoint_heatmap_stream"):
        self.config = load_config()
        self.node_name: str = name
        self.sync: TimeSync

        self.timestamps: List[ros.Time] = []

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.sync = TimeSync(
            ankor_topic="/keypoints_heatmap", unsynced_topics=["/rs2_topic"]
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            packages, timestamp = self.sync.next()
            self.timestamps.append(timestamp)

            img: Image.Image = packages["/rs2_topic"]["pod"].image
            heatmap: np.array = packages["/keypoints_heatmap"]["pod"].array

            img = overlay_heatmap_on_image(img, heatmap)
            img = img.resize((512, 512))
            fps, latency = compute_fps_and_latency(self.timestamps)
            img = add_info_to_image(
                image=img,
                title="Keypoint Heatmap",
                frame_rate=f"{fps} Hz",
                latency=f"{latency} ms",
            )

            cv2.imshow("Keypoint Heatmap", np.array(img)[..., ::-1])
            cv2.waitKey(10)
            self.rate.sleep()


def main():
    node = KeypointHeatmapStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
