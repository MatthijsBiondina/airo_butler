from copy import deepcopy
from PIL import Image
import pickle
from threading import Lock
from typing import List, Optional

import cv2
from pairo_butler.labelling.label_orientation import OrientationLabeler
from pairo_butler.utils.transformation_matrices import TransformationMatrixRS2Sophie
from pairo_butler.plotting.plotting_utils import (
    add_info_to_image,
    compute_fps_and_latency,
)
from pairo_butler.data.timesync import TimeSync
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout
from airo_butler.msg import PODMessage
import numpy as np


class KalmanStream:
    RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "kalman_stream"):
        self.config = load_config()
        self.node_name = name
        self.lock: Lock = Lock()
        self.timestamps: List[ros.Time] = []
        self.kalman_sub: ros.Subscriber
        self.sync: TimeSync

        self.keypoint: Optional[np.ndarray] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.sync = TimeSync(ankor_topic="/rs2_topic", unsynced_topics=["/ur5e_sophie"])
        self.kalman_sub = ros.Subscriber(
            "/kalman_filter_state",
            PODMessage,
            self.__kalman_sub_callback,
            queue_size=self.QUEUE_SIZE,
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            packages, timestamp = self.sync.next()
            self.timestamps.append(timestamp)

            img: Image.Image = packages["/rs2_topic"]["pod"].image

            with self.lock:
                kp = deepcopy(self.keypoint)

            if kp is not None:
                kp_theta = kp[-1]
                kp_coord = np.array(kp[:3])
                sophie_tcp = np.array(packages["/ur5e_sophie"]["pod"].tcp_pose)
                intrinsics_matrix = np.array(
                    packages["/rs2_topic"]["pod"].intrinsics_matrix
                )

                axes = OrientationLabeler.compute_axes(
                    kp_theta, kp_coord, sophie_tcp, intrinsics_matrix
                )
                img = OrientationLabeler.draw_axes_on_image(img, *axes)

            self.render(img)

            self.rate.sleep()

    def render(self, img):
        img = img.resize((512, 512))
        fps, latency = compute_fps_and_latency(self.timestamps)
        img = add_info_to_image(
            image=img,
            title="Grasp Point",
            frame_rate=f"{fps} Hz",
            latency=f"{latency} ms",
        )

        cv2.imshow("Grasp Point", np.array(img)[..., ::-1])
        cv2.waitKey(10)

    def __kalman_sub_callback(self, msg):
        pod = pickle.loads(msg.data)
        means = pod.means
        if means.size == 0:
            with self.lock:
                self.keypoint = None
        else:
            determinants = np.linalg.det(pod.covariances)
            index_most_certain_keypoint = np.argmin(determinants)
            with self.lock:
                self.keypoint = means[index_most_certain_keypoint]


def main():
    node = KalmanStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
