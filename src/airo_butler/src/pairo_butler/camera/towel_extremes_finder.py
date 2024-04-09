from typing import List
from matplotlib.figure import Figure
import numpy as np
from pairo_butler.utils.pods import CoordinatePOD, publish_pod
from pairo_butler.utils.point_cloud_utils import transform_points_to_different_frame
from pairo_butler.camera.zed_camera import ZEDClient
from pairo_butler.utils.tools import UGENT, load_camera_transformation_matrix, pyout
import rospy as ros
from airo_butler.msg import PODMessage
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class TowelExtremesFinder:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 5
    BUFFER_SIZE = 100
    PLOTTING = True

    def __init__(self, name: str = "zed_points_selector"):
        self.node_name = name
        self.transformation_matrix = load_camera_transformation_matrix("T_zed_sophie")
        self.zed: ZEDClient

        self.publisher_max: ros.Publisher
        self.publisher_min: ros.Publisher

        self.high_point_history: List[float] = [
            np.zeros(3) for _ in range(self.BUFFER_SIZE)
        ]
        self.low_point_history: List[float] = [
            np.zeros(3) for _ in range(self.BUFFER_SIZE)
        ]

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.zed = ZEDClient()

        self.publisher_max = ros.Publisher(
            "/towel_top", PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.publisher_min = ros.Publisher(
            "/towel_bot", PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            cloud_in_zed_frame = self.zed.pod.point_cloud[:, :3]

            # transform to world frame
            cloud = transform_points_to_different_frame(
                cloud_in_zed_frame, self.transformation_matrix
            )

            highest_point = self.compute_highest_point(cloud)
            lowest_point = self.compute_lowest_point(cloud)

            self.high_point_history.append(highest_point)
            self.high_point_history.pop(0)
            self.low_point_history.append(lowest_point)
            self.low_point_history.pop(0)

            publish_pod(self.publisher_max, CoordinatePOD(*highest_point.tolist()))
            publish_pod(self.publisher_min, CoordinatePOD(*lowest_point.tolist()))

            if self.PLOTTING:
                self.plot()

    def compute_highest_point(self, cloud: np.ndarray):
        mask = (
            (cloud[:, 2] > 0)
            & (cloud[:, 1] < 0.25)
            & (cloud[:, 1] > -0.25)
            & (cloud[:, 0] > -0.45)
            & (cloud[:, 0] < 0.45)
        )
        try:
            valid_points = cloud[mask]
            idxs = np.argsort(valid_points[:, 2])
            return valid_points[idxs[int(idxs.size * 0.995)]]
        except IndexError:
            return np.zeros(3)
        except ValueError:
            return np.zeros(3)

    def compute_lowest_point(self, cloud: np.ndarray):

        mask = (cloud[:, 2] > 0.03) & (
            np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2) < 0.25
        )
        valid_points = cloud[mask]
        try:
            valid_points = cloud[mask]
            idxs = np.argsort(valid_points[:, 2])
            return valid_points[idxs[int(idxs.size * 0.005)]]
        except IndexError:
            return np.zeros(3)
        except ValueError:
            return np.zeros(3)

    def plot(self):
        # Create a new figure and a subplot
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Data preparation
        max_history_np = np.stack(self.high_point_history, axis=0)[..., 2]
        min_history_np = np.stack(self.low_point_history, axis=0)[..., 2]
        times = range(len(max_history_np))  # Simple range for X axis

        # Plotting
        ax.clear()
        ax.plot(times, max_history_np, color=UGENT.BLUE, label="High")
        ax.plot(times, min_history_np, color=UGENT.ORANGE, label="Low")
        ax.legend()

        # Drawing the canvas
        canvas.draw()

        # Converting Matplotlib figure to OpenCV image
        img = np.array(canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Show the image using OpenCV
        cv2.imshow("Highest & Lowest Towel Point (Z)", img)
        cv2.waitKey(10)


def main():
    node = TowelExtremesFinder()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
