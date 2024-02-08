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

        # Placeholders
        self.means: np.ndarray = np.empty((0, 4))
        self.covariances: np.ndarray = np.empty((0, 4, 4))
        self.timestamps: List[ros.Time] = []

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.pod_client = PODClient("/kalman_filter_state", KalmanFilterStatePOD)

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            pod = self.pod_client.pod

            img = self.__render_pyvista_image(pod.means, pod.covariances)

            cv2.imshow(self.node_name, img)
            cv2.waitKey(10)

            self.rate.sleep()

    def __render_pyvista_image(
        self, means: np.ndarray, covariances: np.ndarray, view_angle=1.5 * np.pi
    ):
        means, covariances, certainties = self.__sort_on_certainty(means, covariances)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_axes()
        for ii, (mean, covariance) in enumerate(zip(means, covariances)):
            point = pv.PolyData(mean[:3].reshape(1, 3))
            color = UGENT.COLORS[ii % len(UGENT.COLORS)]
            opacity = 0.75**ii
            plotter.add_mesh(point, color=color, opacity=opacity, point_size=10)

            eigenvalues, eigenvectors = np.linalg.eigh(covariance[:3, :3])
            radii = np.sqrt(eigenvalues)

            ellipsoid = pv.ParametricEllipsoid(*radii)
            ellipsoid.rotate_vector(
                eigenvectors[:, 0],
                angle=np.rad2deg(np.arccos(eigenvectors[:, 2][2])),
            )
            ellipsoid.rotate_vector(
                eigenvectors[:, 1],
                angle=np.rad2deg(np.arccos(eigenvectors[:, 2][1])),
            )
            ellipsoid.rotate_vector(
                eigenvectors[:, 2],
                angle=np.rad2deg(np.arccos(eigenvectors[:, 2][0])),
            )
            ellipsoid.translate(mean[:3])

            plotter.add_mesh(ellipsoid, color=color, opacity=opacity / 2)

        labels = {"xlabel": "x", "ylabel": "y", "zlabel": "z"}
        plotter.add_axes(**labels)
        camera_position = [3 * np.cos(view_angle), 3 * np.sin(view_angle), 1.0]
        focal_point = [-0.45, 0.0, 0.5]
        view_up = [0, 0, 1]
        plotter.camera_position = [camera_position, focal_point, view_up]

        plotter.show(auto_close=False)
        img_array = plotter.screenshot()
        plotter.close()

        return img_array

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
