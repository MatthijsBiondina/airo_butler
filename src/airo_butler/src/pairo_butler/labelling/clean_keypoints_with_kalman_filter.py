import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Optional
import cv2
import numpy as np

import rospkg
from pairo_butler.utils.pods import (
    BooleanPOD,
    KalmanFilterStatePOD,
    KeypointMeasurementPOD,
    publish_pod,
)
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, pyout
import rospy as ros
import yaml
from airo_butler.msg import PODMessage
from airo_butler.srv import Reset
from PIL import Image, ImageDraw
from airo_butler.srv import PODService

np.set_printoptions(precision=2, suppress=True)


class KeypointCleaner:
    QUEUE_SIZE = 2
    RATE = 30

    def __init__(self, name: str = "keypoint_cleaner"):
        self.node_name = name
        self.pub_name = "/keypoint_measurements"
        self.rate: Optional[ros.Rate] = None

        self.measurement_publisher: Optional[ros.Publisher] = None

        self.config, self.T_sophie_cam = self.__load_config_files()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.measurement_publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):

        for trial in pbar(
            listdir(self.config["folder"]),
            desc="Cleaning",
        ):
            raw_data, valid = self.__load_data_trial(trial)
            if not valid:
                continue
            self.__signal_kalman_reset()
            for measurement, state, frame in pbar(
                zip(
                    raw_data["keypoints"],
                    raw_data["state_sophie"],
                    raw_data["frames"],
                ),
                desc="Filtering",
                total=len(raw_data["keypoints"]),
            ):
                self.__disp_frame(frame, measurement)

                if measurement is None or len(measurement) == 0:
                    continue
                tcp = np.array(state["tcp_pose"]) @ self.T_sophie_cam
                measurement_pod = KeypointMeasurementPOD(
                    timestamp=ros.Time.now(),
                    keypoints=np.array(measurement),
                    camera_tcp=tcp,
                    orientations=np.zeros(len(measurement)),
                    camera_intrinsics=raw_data["rs2_intrinsics"],
                )
                publish_pod(self.measurement_publisher, measurement_pod)

                self.rate.sleep()

            means, covariances = self.__get_kalman_state()
            self.__save(trial, raw_data, means, covariances)

    def __load_config_files(self):
        with open(Path(__file__).parent / "labelling_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        tcp_path = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )
        camera_tcp = np.load(tcp_path / "T_rs2_sophie.npy")
        return config, camera_tcp

    def __load_data_trial(self, path: Path):
        with open(path / "state.json", "r") as f:
            state = json.load(f)
        state["frames"] = load_mp4_video(path / "video.mp4")
        return state, state["valid"]

    def __signal_kalman_reset(self, timeout=10):
        try:
            ros.wait_for_service("reset_kalman_filter", timeout=timeout)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to Kalman Filter server. Is it running?")
            ros.signal_shutdown("Cannot connect to Kalman Filter server.")
            sys.exit(0)
        service = ros.ServiceProxy("reset_kalman_filter", Reset)
        try:
            resp = service()
        except ros.service.ServiceException:
            ros.logwarn(
                f"Service reset_kalman_filter available, but unable to connect."
            )
            return

        if not resp.success:
            ros.logerr(f"Failed to reset Kalman Filter.")
            ros.signal_shutdown("Failed to reset Kalman Filter")
            ros.exit(0)

    def __disp_frame(
        self, frame: Image.Image, keypoints: Optional[np.ndarray] = None, radius=5
    ):
        if keypoints is not None:
            draw = ImageDraw.Draw(frame)
            for keypoint in keypoints:
                # Circle parameters
                center = tuple(keypoint)  # Center of the circle
                circle_bounds = [
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ]
                circle_color = UGENT.GREEN  # Color of the circle

                # Draw the circle
                draw.ellipse(circle_bounds, fill=circle_color)

        cv2.imshow("Data cleaning", np.array(frame)[..., ::-1])
        if cv2.waitKey(10) & 0xFF == ord("q"):
            sys.exit(0)

    def __get_kalman_state(self, service_name: str = "get_kalman_state"):
        try:
            ros.wait_for_service(service_name, timeout=10)
        except ros.exceptions.ROSException:
            ros.logerr(f"Cannot connect to Kalman Filter. Is it running?")
            ros.signal_shutdown("Cannot connect to Kalman Filter.")
            sys.exit(0)

        service = ros.ServiceProxy(service_name, PODService)
        resp = service(pickle.dumps(BooleanPOD(True)))

        state_pod: KalmanFilterStatePOD = pickle.loads(resp.pod)

        return state_pod.means, state_pod.covariances

    def __save(
        self,
        trial: Path,
        data: Dict[str, Any],
        means: np.ndarray,
        covariances: np.ndarray,
    ):
        del data["frames"]

        data["keypoints_world"] = []
        for mean, covariance in zip(means, covariances):
            data["keypoints_world"].append(
                {"mean": mean.tolist(), "covariance": covariance.tolist()}
            )

        with open(trial / "state.json", "w") as f:
            json.dump(data, f)


def main():
    node = KeypointCleaner()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
