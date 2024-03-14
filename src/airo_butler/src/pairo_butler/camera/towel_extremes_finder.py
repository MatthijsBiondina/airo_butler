import numpy as np
from pairo_butler.utils.pods import CoordinatePOD, publish_pod
from pairo_butler.utils.point_cloud_utils import transform_points_to_different_frame
from pairo_butler.camera.zed_camera import ZEDClient
from pairo_butler.utils.tools import load_camera_transformation_matrix, pyout
import rospy as ros
from airo_butler.msg import PODMessage


class TowelExtremesFinder:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 5

    def __init__(self, name: str = "zed_points_selector"):
        self.node_name = name
        self.transformation_matrix = load_camera_transformation_matrix("T_zed_sophie")
        self.zed: ZEDClient

        self.publisher_max: ros.Publisher
        self.publisher_min: ros.Publisher

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

            publish_pod(self.publisher_max, CoordinatePOD(*highest_point.tolist()))
            publish_pod(self.publisher_min, CoordinatePOD(*lowest_point.tolist()))

    def compute_highest_point(self, cloud: np.ndarray):
        mask = (
            (cloud[:, 2] > 0)
            & (cloud[:, 1] < 0.35)
            & (cloud[:, 1] > -0.35)
            & (cloud[:, 0] > -0.45)
            & (cloud[:, 0] < 0.45)
        )
        try:
            valid_points = cloud[mask]
            idx = np.argmax(valid_points[:, 2])

            return valid_points[idx]
        except IndexError:
            return np.zeros(3)

    def compute_lowest_point(self, cloud: np.ndarray):

        mask = (cloud[:, 2] > 0.02) & (
            np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2) < 0.15  # cone
        )
        valid_points = cloud[mask]
        try:
            valid_points = cloud[mask]
            idx = np.argmin(valid_points[:, 2])

            return valid_points[idx]
        except IndexError:
            return np.zeros(3)


def main():
    node = TowelExtremesFinder()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
