from pathlib import Path
import numpy as np
from airo_butler.msg import PODMessage
import rospkg
from pairo_butler.utils.pods import KeypointMeasurementPOD, publish_pod
from pairo_butler.data.timesync import TimeSync
import rospy as ros
from pairo_butler.utils.tools import load_config, pyout


class KalmanMeasurementCollector:
    RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "kalman_measurement_collector"):
        self.node_name: str = name
        self.rate: ros.Rate
        self.publisher: ros.Publisher
        self.sync: TimeSync
        self.T_sophie_rs2 = self.__init_camera_transformation_matrix()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.rate = ros.Rate(self.RATE)
        self.sync = TimeSync(
            ankor_topic="/keypoints_theta",
            unsynced_topics=[
                "/rs2_topic",
                "/keypoints_uv",
                "/keypoints_theta",
                "/ur5e_sophie",
            ],
        )
        self.publisher = ros.Publisher(
            "/keypoint_measurements", PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            packages, timestamp = self.sync.next()

            keypoint_uv = packages["/keypoints_uv"]["pod"]
            keypoint_theta = packages["/keypoints_theta"]["pod"]

            if not (keypoint_uv.valid and keypoint_theta.valid):
                continue

            pod = KeypointMeasurementPOD(
                timestamp=timestamp,
                keypoints=np.array([keypoint_uv.x, keypoint_uv.y])[None, :],
                camera_tcp=packages["/ur5e_sophie"]["pod"].tcp_pose @ self.T_sophie_rs2,
                orientations=np.array([keypoint_theta.mean]),
                camera_intrinsics=packages["/rs2_topic"]["pod"].intrinsics_matrix,
            )

            publish_pod(self.publisher, pod)

            self.rate.sleep()

    def __init_camera_transformation_matrix(self):
        tcp_path = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )
        camera_tcp = np.load(tcp_path / "T_rs2_tcp_sophie.npy")
        return camera_tcp


def main():
    node = KalmanMeasurementCollector()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
