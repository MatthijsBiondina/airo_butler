import json
from pathlib import Path
import sys
from typing import Optional
import numpy as np

import rospkg
from pairo_butler.utils.pods import KeypointMeasurementPOD, publish_pod
from pairo_butler.utils.tools import listdir, pbar, pyout
import rospy as ros
import yaml
from airo_butler.msg import PODMessage
from airo_butler.srv import Reset

np.set_printoptions(precision=2, suppress=True)


class KeypointCleaner:
    QUEUE_SIZE = 2
    RATE = 30

    def __init__(self, name: str = "keypoint_cleaner"):
        self.node_name = name
        self.pub_name = "/keypoint_measurements"
        self.rate: Optional[ros.Rate] = None

        self.publisher: Optional[ros.Publisher] = None

        self.config, self.T_sophie_cam = self.__load_config_files()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.publisher = ros.Publisher(
            self.pub_name, PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():  # todo: debug rem
            for trial in pbar(listdir(self.config["folder"]), desc="Cleaning"):
                trial = list(listdir(self.config["folder"]))[0]

                raw_data, valid = self.__load_data_trial(trial)
                if not valid:
                    continue
                self.__signal_kalman_reset()
                for measurement, state in pbar(
                    zip(raw_data["keypoints"], raw_data["state_sophie"]),
                    desc="Filtering",
                    total=len(raw_data["keypoints"]),
                ):
                    if measurement is None:
                        continue
                    tcp = np.array(state["tcp_pose"]) @ self.T_sophie_cam
                    measurement_pod = KeypointMeasurementPOD(
                        timestamp=ros.Time.now(),
                        keypoints=np.array(measurement),
                        camera_tcp=tcp,
                        orientations=np.zeros(len(measurement)),
                        camera_intrinsics=raw_data["rs2_intrinsics"],
                    )
                    publish_pod(self.publisher, measurement_pod)
                    self.rate.sleep()

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


def main():
    node = KeypointCleaner()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
