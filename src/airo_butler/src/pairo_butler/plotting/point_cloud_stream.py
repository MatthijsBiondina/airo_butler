from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pyvista as pv
import rospkg
from PIL import Image
from pairo_butler.plotting.plotting_utils import add_info_to_image
from pairo_butler.utils.point_cloud_utils import (
    rgb_to_hue,
    transform_points_to_different_frame,
)
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros


class PointCloudStream:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 10
    SIZE = (768, 512)

    def __init__(self, name: str = "point_cloud_stream") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.zed: Optional[ZEDClient] = None

        self.tranformation_matrix_sophie_zed: np.ndarray = (
            self.__load_transformation_matrix()
        )
        self.view_angle = 1.5 * np.pi

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.zed = ZEDClient()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            self.tranformation_matrix_sophie_zed = self.__load_transformation_matrix()

            cloud = self.zed.pod.point_cloud

            xyz_in_zed_frame, points_rgb_colors = cloud[:, :3], cloud[:, 3:]

            xyz_in_sophie_frame = transform_points_to_different_frame(
                xyz_in_zed_frame, self.tranformation_matrix_sophie_zed
            )

            point_cloud_image = self.__render_pyvista_image(
                xyz_in_sophie_frame, points_rgb_colors
            )
            image = self.__add_info_to_image(point_cloud_image)

            cv2.imshow("ZED2i (Point-Cloud)", np.array(image)[..., ::-1])
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == ord("q"):
                ros.signal_shutdown("Visualization window closed by user.")
                break
            elif key_pressed == ord("a"):
                self.view_angle = (self.view_angle - np.deg2rad(3)) % (2 * np.pi)
            elif key_pressed == ord("d"):
                self.view_angle = (self.view_angle + np.deg2rad(3)) % (2 * np.pi)

            self.rate.sleep()

    def __load_transformation_matrix(self) -> np.ndarray:
        fpath = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "res"
            / "camera_tcps"
            / "T_zed_sophie.npy"
        )

        transformation_matrix = np.load(fpath)
        return transformation_matrix

    def __render_pyvista_image(
        self, points: np.ndarray, colors: np.ndarray
    ) -> np.ndarray:
        point_cloud = pv.PolyData(points)
        point_cloud["colors"] = (colors).astype(np.uint8)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_points(point_cloud, scalars="colors", rgb=True)
        labels = {"xlabel": "x", "ylabel": "y", "zlabel": "z"}
        plotter.add_axes(**labels)

        camera_position = [
            3 * np.cos(self.view_angle),
            3 * np.sin(self.view_angle),
            0.1,
        ]

        focal_point = [0.0, 0.0, 0.1]
        view_up = [0, 0, 1]
        plotter.camera_position = [camera_position, focal_point, view_up]

        plotter.show(auto_close=False)
        img_array = plotter.screenshot()
        plotter.close()

        return img_array

    def __add_info_to_image(self, image: np.ndarray) -> Image:
        frame = Image.fromarray(np.copy(image))

        frame = frame.resize(self.SIZE)
        frame = add_info_to_image(
            frame,
            title="ZED2i (Point-Cloud)",
            frame_rate=f"{self.zed.fps} Hz",
            latency=f"{self.zed.latency} ms",
        )

        return Image.fromarray(np.array(frame))


def main():
    node = PointCloudStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
