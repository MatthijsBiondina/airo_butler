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


class OrientationLabeler:
    RATE = 60

    def __init__(self, name: str = "orientation_labeler") -> None:
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
        self.rate = ros.Rate(self.RATE)
        ros.on_shutdown(cv2.destroyAllWindows)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self) -> None:
        for ii, trial in enumerate(listdir(self.config["folder"])):
            if ii < 500:
                continue

            if ros.is_shutdown():
                break

            ros.loginfo(trial)

            data, valid = LabellingUtils.load_trial_data(trial)
            if not valid:
                continue

            data = self.__labelling_gui(data)

            self.__save_data(trial, data)

    def __labelling_gui(self, data: Dict[str, Any], dtheta=1):
        nr_of_frames = len(data["frames"])

        frame_idx = 0
        for kp_idx, keypoint_world in enumerate(data["keypoints_world"]):
            kp_coord = keypoint_world["mean"][:3]
            kp_theta = keypoint_world["mean"][3]

            while True:
                origin, x_axis, y_axis, z_axis = self.__compute_axes(
                    kp_theta=kp_theta,
                    kp_coord=kp_coord,
                    sophie_tcp=np.array(data["state_sophie"][frame_idx]["tcp_pose"]),
                    intrinsics_matrix=np.array(data["rs2_intrinsics"]),
                )
                frame = data["frames"][frame_idx].copy()
                frame = self.__draw_axes_on_image(frame, origin, x_axis, y_axis, z_axis)
                user_input = self.__show_frame_for_labelling(frame)

                if user_input == ord("a"):
                    frame_idx = np.clip(frame_idx - 1, a_min=0, a_max=nr_of_frames - 1)
                elif user_input == ord("d"):
                    frame_idx = np.clip(frame_idx + 1, a_min=0, a_max=nr_of_frames - 1)
                elif user_input == ord("q"):
                    kp_theta = (kp_theta - np.deg2rad(dtheta) + np.pi) % (
                        2 * np.pi
                    ) - np.pi
                elif user_input == ord("e"):
                    kp_theta = (kp_theta + np.deg2rad(dtheta) + np.pi) % (
                        2 * np.pi
                    ) - np.pi
                elif user_input == 32:  # space bar
                    break
                elif user_input == 255:
                    pass
                else:
                    ros.logwarn(f"Unknown input key: {user_input}")

                self.rate.sleep()

            data["keypoints_world"][kp_idx]["mean"][3] = kp_theta

        return data

    def __compute_axes(
        self,
        kp_theta: float,
        kp_coord: np.ndarray,
        sophie_tcp: np.ndarray,
        intrinsics_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        camera_tcp = sophie_tcp @ self.T_sophie_cam

        # Construct transformation matrix
        T = np.array(
            [
                [np.cos(kp_theta), -np.sin(kp_theta), 0, kp_coord[0]],
                [np.sin(kp_theta), np.cos(kp_theta), 0, kp_coord[1]],
                [0.0, 0.0, 1.0, kp_coord[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        axs = self.config["axis_length"]  # axis length

        origin_world = T @ np.array([0.0, 0.0, 0.0, 1.0])[:, None]
        x_axis_world = T @ np.array([0.0, -axs, 0.0, 1.0])[:, None]
        y_axis_world = T @ np.array([0.0, 0.0, -axs, 1.0])[:, None]
        z_axis_world = T @ np.array([axs, 0.0, 0.0, 1.0])[:, None]

        # note that these are (4x1) with the last index 1. This index corresponds
        # with orientation as far as the kalman filter is concerned, the output
        # of which we ignore here. So we pass the world coordinates with the trailing
        # 1 implicitly
        origin_frame = KalmanFilter.calculate_expected_measurements(
            keypoint_world=origin_world,
            camera_tcp=camera_tcp,
            camera_intrinsics=intrinsics_matrix,
        )
        x_axis_frame = KalmanFilter.calculate_expected_measurements(
            keypoint_world=x_axis_world,
            camera_tcp=camera_tcp,
            camera_intrinsics=intrinsics_matrix,
        )
        y_axis_frame = KalmanFilter.calculate_expected_measurements(
            keypoint_world=y_axis_world,
            camera_tcp=camera_tcp,
            camera_intrinsics=intrinsics_matrix,
        )
        z_axis_frame = KalmanFilter.calculate_expected_measurements(
            keypoint_world=z_axis_world,
            camera_tcp=camera_tcp,
            camera_intrinsics=intrinsics_matrix,
        )

        origin = origin_frame[:2].squeeze(-1)
        x_axis = x_axis_frame[:2].squeeze(-1)
        y_axis = y_axis_frame[:2].squeeze(-1)
        z_axis = z_axis_frame[:2].squeeze(-1)

        return origin, x_axis, y_axis, z_axis

    def __draw_axes_on_image(
        self,
        image: Image,
        origin: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        width: int = 3,
    ) -> Image:
        draw = ImageDraw.Draw(image)

        draw.line(
            (origin[0], origin[1], x_axis[0], x_axis[1]), fill=UGENT.RED, width=width
        )
        draw.line(
            (origin[0], origin[1], y_axis[0], y_axis[1]), fill=UGENT.GREEN, width=width
        )
        draw.line(
            (origin[0], origin[1], z_axis[0], z_axis[1]), fill=UGENT.BLUE, width=width
        )

        return image

    def __show_frame_for_labelling(self, frame: PIL.Image) -> int:
        cv2.imshow("Label Orientation", np.array(frame)[..., ::-1])
        key = cv2.waitKey(10) & 0xFF
        return key

    def __save_data(self, path: Path, data: Dict[str, Any]) -> None:
        if "frames" in data:
            del data["frames"]

        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = val.tolist()

        with open(path / "state.json", "w") as f:
            json.dump(data, f, indent=2)


def main():
    node = OrientationLabeler()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
