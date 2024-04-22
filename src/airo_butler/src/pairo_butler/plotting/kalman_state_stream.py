import pickle
from threading import Lock
from typing import List, Optional

import cv2
from pairo_butler.utils.pod_client import PODClient
from pairo_butler.utils.pods import KalmanFilterStatePOD
import pyvista as pv
from pairo_butler.utils.tools import UGENT, pyout
import numpy as np
import rospy as ros
from airo_butler.msg import PODMessage

np.set_printoptions(precision=2, suppress=True)


class KalmanStream:
    QUEUE_SIZE = 2
    RATE = 30
    SIZE = (512, 512)

    def __init__(self, name: str = "kalman_filter_visualization") -> None:
        self.node_name = name
        self.rate: Optional[ros.Rate] = None
        self.lock: Lock = Lock()

        self.subscriber: Optional[ros.Subscriber] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.pod_client = PODClient(
            "/kalman_filter_state", KalmanFilterStatePOD, timeout=-1
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            pod = self.pod_client.pod
            if pod.means.size == 0:
                self.rate.sleep()
                continue

            img = self.__render_pyvista_image(
                pod.means, pod.covariances, pod.camera_tcp
            )

            cv2.imshow(self.node_name, img)
            cv2.waitKey(10)

            self.rate.sleep()

    def __render_pyvista_image(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        camera_tcp: np.ndarray,
        view_angle=-0.25 * np.pi,
    ):
        means, covariances, certainties = self.__sort_on_certainty(means, covariances)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_axes(line_width=5, color=UGENT.BLACK)
        box = pv.Box(bounds=[-0.45, 0.45, -0.45, 0.45, 0.0, 1.0])
        plotter.add_mesh(box, color=UGENT.BLACK, line_width=2, style="wireframe")

        for ii, (mean, covariance) in enumerate(zip(means, covariances)):

            color = UGENT.COLORS[ii % len(UGENT.COLORS)]
            base_opacity = 0.75**ii

            plotter.add_mesh(
                pv.PolyData(mean[:3].reshape(1, 3)),
                color=color,
                opacity=base_opacity,
                point_size=10,
            )
            plotter.add_mesh(
                pv.PolyData(np.concatenate((mean[:2], np.array([0.0]))).reshape(1, 3)),
                color=color,
                opacity=base_opacity / 2,
                point_size=10,
            )

            eigenvalues, eigenvectors = np.linalg.eigh(covariance[:3, :3])
            radii = np.sqrt(eigenvalues)

            sphere = pv.Sphere(radius=1, center=[0, 0, 0])
            ellipsoid = sphere.scale(radii)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = eigenvectors
            transformation_matrix[:3, 3] = mean[:3]
            ellipsoid = ellipsoid.transform(transformation_matrix)
            plotter.add_mesh(ellipsoid, color=color, opacity=base_opacity / 2)

        self.__draw_camera(plotter, camera_tcp)

        labels = {"xlabel": "x", "ylabel": "y", "zlabel": "z"}
        plotter.add_axes(**labels)
        camera_position = [3 * np.cos(view_angle), 3 * np.sin(view_angle), 1.0]
        focal_point = [-0.45, 0.0, 0.5]
        view_up = [0, 0, 1]
        plotter.camera_position = [camera_position, focal_point, view_up]

        plotter.show(auto_close=False)
        img_array = plotter.screenshot()
        plotter.close()

        return img_array[..., ::-1]

    def __draw_camera(self, plotter, camera_tcp: np.ndarray, axis_length=0.1):
        point = pv.PolyData(camera_tcp[:3, 3].reshape(1, 3))
        plotter.add_mesh(point, color=UGENT.BLUE, opacity=1, point_size=10)
        point = pv.PolyData(
            np.concatenate((camera_tcp[:2, 3], np.array([0.0]))).reshape(1, 3)
        )
        plotter.add_mesh(point, color=UGENT.BLUE, opacity=0.5, point_size=10)

        x_axis = pv.Line(
            camera_tcp[:3, 3], camera_tcp[:3, 3] + axis_length * camera_tcp[:3, 0]
        )
        y_axis = pv.Line(
            camera_tcp[:3, 3], camera_tcp[:3, 3] + axis_length * camera_tcp[:3, 1]
        )
        z_axis = pv.Line(
            camera_tcp[:3, 3], camera_tcp[:3, 3] + axis_length * camera_tcp[:3, 2]
        )
        plotter.add_mesh(x_axis, color=UGENT.RED, line_width=5)
        plotter.add_mesh(y_axis, color=UGENT.GREEN, line_width=5)
        plotter.add_mesh(z_axis, color=UGENT.BLUE, line_width=5)

    def __sort_on_certainty(self, means: np.ndarray, covariances: np.ndarray):
        uncertainties = np.linalg.det(covariances[:])
        sorted_indices = np.argsort(uncertainties)

        sorted_means = means[sorted_indices]
        sorted_covariances = covariances[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]

        relative_certainty = sorted_uncertainties[0] / sorted_uncertainties

        return sorted_means, sorted_covariances, relative_certainty


def main():
    node = KalmanStream()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
