from pathlib import Path
from typing import List, Optional
import numpy as np
import rospkg
from pairo_butler.labelling.label_orientation import OrientationLabeler
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.pods import KeypointThetaPOD, KeypointUVPOD
from pairo_butler.plotting.plotting_utils import (
    add_info_to_image,
    compute_fps_and_latency,
    overlay_heatmap_on_image,
)
from pairo_butler.data.timesync import TimeSync
import rospy as ros
from pairo_butler.utils.tools import UGENT, load_config, pyout
from PIL import Image, ImageDraw
import cv2


class KeypointHeatmapStream:
    RATE = 30

    def __init__(self, name: str = "keypoint_heatmap_stream"):
        self.config = load_config()
        self.node_name: str = name
        self.sync: TimeSync

        self.timestamps: List[ros.Time] = []

        self.T_sophie_rs2 = self.__init_camera_transformation_matrix()
        self.kalman_filter = KalmanFilter()
        self.orientation_labeler = OrientationLabeler()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.sync = TimeSync(
            ankor_topic="/keypoints_heatmap",
            unsynced_topics=[
                "/rs2_topic",
                "/keypoints_uv",
                "/keypoints_theta",
                "/ur5e_sophie",
            ],
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            packages, timestamp = self.sync.next()
            self.timestamps.append(timestamp)

            img: Image.Image = packages["/rs2_topic"]["pod"].image
            camera_intrinsics: np.ndarray = np.array(
                packages["/rs2_topic"]["pod"].intrinsics_matrix
            )
            heatmap: np.array = packages["/keypoints_heatmap"]["pod"].array
            kp_uv: KeypointUVPOD = packages["/keypoints_uv"]["pod"]
            kp_theta: KeypointThetaPOD = packages["/keypoints_theta"]["pod"]
            sophie_tcp: np.ndarray = np.array(packages["/ur5e_sophie"]["pod"].tcp_pose)

            heatmap = np.max(heatmap, axis=0)

            img = overlay_heatmap_on_image(img, heatmap)
            self.draw_keypoint(
                img, kp_uv, kp_theta, sophie_tcp, camera_intrinsics, UGENT.BLUE
            )

            self.render(img)

            self.rate.sleep()

    def draw_keypoint(
        self,
        img: Image.Image,
        uv: KeypointUVPOD,
        theta: KeypointThetaPOD,
        sophie_tcp: np.ndarray,
        camera_intrinsics: np.ndarray,
        color=UGENT.BLUE,
    ):
        if uv.valid and theta.valid:
            kf = self.kalman_filter
            camera_tcp = sophie_tcp @ self.T_sophie_rs2
            measurement = np.array([uv.x, uv.y, theta.mean])[:, None]
            kp_world, _ = kf.kalman_measurement_update(
                measurement=measurement,
                camera_tcp=camera_tcp,
                camera_intrinsics=camera_intrinsics,
                sensor_fusion=False,
                n_iterations=5,
            )

            origin, x_axis, y_axis, z_axis = self.orientation_labeler.compute_axes(
                kp_theta=float(kp_world[-1]),
                kp_coord=kp_world[:3].squeeze(1),
                sophie_tcp=sophie_tcp,
                intrinsics_matrix=camera_intrinsics,
            )

            draw = ImageDraw.Draw(img)

            # draw origin
            radius = 5
            left_up_point = (uv.x - radius, uv.y - radius)
            right_down_point = (uv.x + radius, uv.y + radius)
            draw.ellipse([left_up_point, right_down_point], fill=color)

            # draw x-, y-, and z-axis
            draw.line(
                (origin[0], origin[1], x_axis[0], x_axis[1]), fill=UGENT.RED, width=3
            )
            draw.line(
                (origin[0], origin[1], y_axis[0], y_axis[1]), fill=UGENT.GREEN, width=3
            )
            draw.line(
                (origin[0], origin[1], z_axis[0], z_axis[1]), fill=UGENT.BLUE, width=3
            )

    def render(self, img: np.ndarray):
        img = img.resize((512, 512))
        fps, latency = compute_fps_and_latency(self.timestamps)
        img = add_info_to_image(
            image=img,
            title="Computer Vision",
            frame_rate=f"{fps} Hz",
            latency=f"{latency} ms",
        )

        cv2.imshow("Computer Vision", np.array(img)[..., ::-1])
        cv2.waitKey(10)

    def __init_camera_transformation_matrix(self):
        tcp_path = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )
        camera_tcp = np.load(tcp_path / "T_rs2_tcp_sophie.npy")
        return camera_tcp


def main():
    node = KeypointHeatmapStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
