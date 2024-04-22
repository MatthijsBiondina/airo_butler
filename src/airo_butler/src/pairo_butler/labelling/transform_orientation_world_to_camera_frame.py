import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import PIL
import cv2
import numpy as np
import rospkg
import yaml
from pairo_butler.labelling.labelling_utils import LabellingUtils
from pairo_butler.labelling.determine_visibility import VisibilityChecker
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, pyout
import rospy as ros


class OrientationFrameTransformer:
    def __init__(self, name: str = "orientation_frame_transformer") -> None:
        self.node_name: str = name

        config_path: Path = Path(__file__).parent / "labelling_config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        matrix_path: Path = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "res"
            / "camera_tcps"
            / "T_rs2_tcp_sophie.npy"
        )
        self.T_sophie_cam: np.ndarray = np.load(matrix_path)

    def start_ros(self) -> None:
        ros.init_node(self.node_name, log_level=ros.INFO)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self) -> None:
        for trial in listdir(self.config["folder"]):
            if ros.is_shutdown():
                break
            ros.loginfo(trial)

            data, valid = LabellingUtils.load_trial_data(trial, include_video=False)
            if not valid:
                continue

            data = self.__transform_orientation_frame(data)

            LabellingUtils.save_trial_data(path=trial, data=data)

    def __transform_orientation_frame(self, data: Dict[str, Any]) -> Dict[str, Any]:
        nr_of_frames = len(data["keypoints_clean"])
        data["keypoints_theta"] = [[] for _ in range(nr_of_frames)]

        for frame_idx in range(nr_of_frames):
            for kp_idx, keypoint_world in enumerate(data["keypoints_world"]):
                kp_coord = keypoint_world["mean"][:3]
                kp_theta = keypoint_world["mean"][3]

                kp_theta_camera = self.__compute_orientation_in_camera_frame(
                    kp_theta=kp_theta,
                    kp_coord=kp_coord,
                    sophie_tcp=np.array(data["state_sophie"][frame_idx]["tcp_pose"]),
                    intrinsics_matrix=np.array(data["rs2_intrinsics"]),
                )

                data["keypoints_theta"][frame_idx].append(kp_theta_camera)

        return data

    def __compute_orientation_in_camera_frame(
        self,
        kp_theta: float,
        kp_coord: np.ndarray,
        sophie_tcp: np.ndarray,
        intrinsics_matrix: np.ndarray,
    ):
        camera_tcp = sophie_tcp @ self.T_sophie_cam

        T = np.array(
            [
                [np.cos(kp_theta), -np.sin(kp_theta), 0, kp_coord[0]],
                [np.sin(kp_theta), np.cos(kp_theta), 0, kp_coord[1]],
                [0.0, 0.0, 1.0, kp_coord[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        origin_world = T @ np.array([0.0, 0.0, 0.0, 1.0])[:, None]

        origin_world[3, 0] = kp_theta

        measured_kp = KalmanFilter.calculate_expected_measurements(
            keypoint_world=origin_world,
            camera_tcp=camera_tcp,
            camera_intrinsics=intrinsics_matrix,
        )

        return measured_kp[2, 0]


def main():
    node = OrientationFrameTransformer()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
