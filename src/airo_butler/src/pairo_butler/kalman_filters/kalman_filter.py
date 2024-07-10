from pathlib import Path
import pickle
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
import yaml
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.utils.tools import pyout
from pairo_butler.utils.pods import (
    BooleanPOD,
    KalmanFilterStatePOD,
    KeypointMeasurementPOD,
    publish_pod,
)
import rospy as ros
from airo_butler.msg import PODMessage
from airo_butler.srv import PODService, Reset

np.set_printoptions(precision=3, suppress=True)


STATE_SIZE = 4


class KalmanFilterClient:
    def __init__(self):
        self.subscriber: ros.Subscriber = ros.Subscriber(
            "/kalman_filter_state", PODMessage, self.__sub_callback, queue_size=2
        )

        self.state = None

    def __sub_callback(self, msg):
        self.state = pickle.loads(msg.data)


class KalmanFilter:
    QUEUE_SIZE = 2
    RATE = 120

    def __init__(self, name: str = "kalman_filter"):
        self.name = name
        self.rate: Optional[ros.Rate] = None
        self.lock: Lock = Lock()
        self.paused: bool = True

        # Subscribers and Publishers
        self.subscriber: Optional[ros.Subscriber] = None
        self.publisher: Optional[ros.Publisher] = None

        # Services
        self.reset_service: Optional[ros.Service] = None
        self.getter_service: Optional[ros.Service] = None

        # Placeholders
        self.pending: List[KeypointMeasurementPOD] = []
        self.mean: np.ndarray = np.empty((0, 1))
        self.covariance: np.ndarray = np.empty((0, 0))
        self.all_camera_tcps: np.ndarray = np.empty((0, 4, 4))

        with open(Path(__file__).parent / "kalman_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.subscriber = ros.Subscriber(
            "/keypoint_measurements",
            PODMessage,
            self.__sub_callback,
            queue_size=self.QUEUE_SIZE,
        )

        self.publisher = ros.Publisher(
            "/kalman_filter_state", PODMessage, queue_size=self.QUEUE_SIZE
        )

        self.reset_service = ros.Service(
            f"reset_{self.name}", Reset, self.__reset_service_callback
        )
        self.getter_service = ros.Service(
            f"get_kalman_state", PODService, self.__getter_service_callback
        )
        self.pause_service = ros.Service(
            f"pause_{self.name}", Reset, self.__pause_service_callback
        )
        self.unpause_service = ros.Service(
            f"unpause_{self.name}", Reset, self.__unpause_service_callback
        )

        ros.loginfo(f"{self.name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            while len(self.pending) == 0:
                self.rate.sleep()

            try:
                with self.lock:
                    try:
                        camera_tcp, measurements, camera_intrinsics, timestamp = (
                            self.__unpack_measurement_pod(self.pending[0])
                        )
                    except IndexError:
                        continue
                for measurement in measurements:
                    self.kalman_measurement_update(
                        measurement, camera_tcp, camera_intrinsics
                    )
                with self.lock:
                    try:
                        self.pending.pop(0)
                    except IndexError:
                        continue
                pod = self.__build_state_pod_message(
                    timestamp=timestamp, camera_tcp=camera_tcp
                )
                publish_pod(self.publisher, pod)
            except Exception as e:
                ros.logwarn(f"Kalman filter: an exception occurred: {e}")

            self.rate.sleep()

    def __reset_service_callback(self, req):
        start_time = ros.Time.now()
        ros.loginfo("Starting Reset Kalman Filter")

        with self.lock:
            ros.loginfo(
                f"Lock acquired after: {(ros.Time.now() - start_time).to_sec()} seconds"
            )
            self.pending = []
            ros.loginfo(
                f"Cleared pending after: {(ros.Time.now() - start_time).to_sec()} seconds"
            )

            ros.loginfo(
                f"mean: {self.mean.size}, "
                f"cov: {self.covariance.size}, "
                f"cams: {self.all_camera_tcps.size}"
            )

            self.mean = np.empty((0, 1))
            self.covariance = np.empty((0, 0))
            self.all_camera_tcps = np.empty((0, 4, 4))
            ros.loginfo(
                f"Cleared data after: {(ros.Time.now() - start_time).to_sec()} seconds"
            )

            pod = self.__build_state_pod_message(
                timestamp=ros.Time.now(), camera_tcp=None
            )
            ros.loginfo(
                f"Initialized data structures after: {(ros.Time.now() - start_time).to_sec()} seconds"
            )

            publish_pod(self.publisher, pod)
            ros.loginfo(
                f"Published new state after: {(ros.Time.now() - start_time).to_sec()} seconds"
            )

        ros.loginfo(
            f"Reset completed in total: {(ros.Time.now() - start_time).to_sec()} seconds"
        )
        return True

    def __getter_service_callback(self, req):
        pod = self.__build_state_pod_message(timestamp=ros.Time.now(), camera_tcp=None)

        return pickle.dumps(pod)

    def __pause_service_callback(self, req):
        self.paused = True
        return True

    def __unpause_service_callback(self, req):
        self.paused = False
        return True

    def __sub_callback(self, msg):
        try:
            if not self.paused:
                self.pending.append(pickle.loads(msg.data))
        except Exception as e:
            ros.logwarn(f"An exception occurred while adding new measurement: {e}")

    def __unpack_measurement_pod(self, pod):
        self.all_camera_tcps = np.concatenate(
            (self.all_camera_tcps, pod.camera_tcp[None, ...]), axis=0
        )
        camera_tcp = pod.camera_tcp

        A = np.concatenate((pod.keypoints, pod.orientations[:, None]), axis=1)

        measurements = np.concatenate(
            (pod.keypoints, pod.orientations[:, None]), axis=1
        )[..., None]
        return camera_tcp, measurements, np.array(pod.camera_intrinsics), pod.timestamp

    def kalman_measurement_update(
        self,
        measurement: np.ndarray,
        camera_tcp: np.ndarray,
        camera_intrinsics: np.ndarray,
        n_iterations: int = 3,
        sensor_fusion: bool = True,
    ):

        measurement_noise_covariance = self.make_Q_matrix()
        new_mean, prior_covariance = self.init_new_mu_and_Sigma()

        for _ in range(n_iterations):
            predicted_measurement = KalmanFilter.calculate_expected_measurements(
                new_mean, camera_tcp, camera_intrinsics
            )
            measurement_matrix = KalmanFilter.calculate_measurement_jacobian(
                new_mean, camera_tcp, camera_intrinsics
            )

            new_mean, new_covariance = KalmanFilter.kalman_update_formula(
                new_mean,
                prior_covariance,
                measurement,
                predicted_measurement,
                measurement_matrix,
                measurement_noise_covariance,
            )

        if sensor_fusion:
            self.__sensor_fusion(new_mean, new_covariance)
        return new_mean, new_covariance

    def make_Q_matrix(self):
        Q = np.array(
            [
                self.config["variance_camera_plane"],
                self.config["variance_camera_plane"],
                self.config["variance_orientation_measurement"],
            ]
        )
        Q = np.diag(Q)
        return Q

    def init_new_mu_and_Sigma(self, eps=1e-3):
        # It is important that we do not initialize at (0,0,0) for numerical stability
        new_mean = np.array([0.0, 0.0, 0.5, 0.0])[:, None]
        new_covariance = np.array(
            [
                self.config["variance_position_prior"],
                self.config["variance_position_prior"],
                self.config["variance_position_prior"],
                self.config["variance_orientation_prior"],
            ]
        )
        new_covariance = np.diag(new_covariance)
        return new_mean, new_covariance

    @staticmethod
    def calculate_expected_measurements(
        keypoint_world: np.ndarray,
        camera_tcp: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(keypoint_world, np.ndarray)
        assert isinstance(camera_tcp, np.ndarray)
        assert isinstance(camera_intrinsics, np.ndarray)
        assert keypoint_world.shape == (4, 1)
        assert camera_tcp.shape == (4, 4)
        assert camera_intrinsics.shape == (3, 3)

        keypoint_xyz_world = keypoint_world[:3]
        camera_xyz_world = camera_tcp[:3, 3][..., None]
        camera_rotation_matrix = np.linalg.inv(camera_tcp[:3, :3])
        keypoint_xyz_camera = camera_rotation_matrix @ (
            keypoint_xyz_world - camera_xyz_world
        )
        camera_focal_length_x = camera_intrinsics[0, 0]
        camera_focal_length_y = camera_intrinsics[1, 1]
        camera_principal_point_x = camera_intrinsics[0, 2]
        camera_principal_point_y = camera_intrinsics[1, 2]

        expected_keypoint_x = (
            camera_focal_length_x * keypoint_xyz_camera[0] / keypoint_xyz_camera[2]
            + camera_principal_point_x
        )
        expected_keypoint_y = (
            camera_focal_length_y * keypoint_xyz_camera[1] / keypoint_xyz_camera[2]
            + camera_principal_point_y
        )

        keypoint_angle_world = keypoint_world[3]
        camera_z_projection_on_xy_plane = -camera_tcp[:2, 2] / np.linalg.norm(
            camera_tcp[:2, 2]
        )
        camera_angle_world = np.arctan2(
            camera_z_projection_on_xy_plane[1], camera_z_projection_on_xy_plane[0]
        )
        expected_keypoint_angle = (
            keypoint_angle_world - camera_angle_world + np.pi
        ) % (2 * np.pi) - np.pi

        return np.array(
            [
                expected_keypoint_x,
                expected_keypoint_y,
                expected_keypoint_angle,
            ]
        )

    @staticmethod
    def calculate_measurement_jacobian(keypoint_world, camera_tcp, camera_intrinsics):
        keypoint_xyz_world = keypoint_world[:3]
        camera_xyz_workd = camera_tcp[:3, 3][..., None]
        camera_rotation_matrix = np.linalg.inv(camera_tcp[:3, :3])
        keypoint_xyz_camera = camera_rotation_matrix @ (
            keypoint_xyz_world - camera_xyz_workd
        )
        camera_focal_length_x = camera_intrinsics[0, 0]
        camera_focal_length_y = camera_intrinsics[1, 1]

        gradient_pixel_x = camera_focal_length_x * (
            camera_rotation_matrix[0] / keypoint_xyz_camera[2]
            - camera_rotation_matrix[2]
            * keypoint_xyz_camera[0]
            / keypoint_xyz_camera[2] ** 2
        )
        gradient_pixel_y = camera_focal_length_y * (
            camera_rotation_matrix[1] / keypoint_xyz_camera[2]
            - camera_rotation_matrix[2]
            * keypoint_xyz_camera[1]
            / keypoint_xyz_camera[2] ** 2
        )
        gradient_angle = 1.0

        jacobian = np.zeros((3, keypoint_world.shape[0]))
        jacobian[0, :3] = gradient_pixel_x
        jacobian[1, :3] = gradient_pixel_y
        jacobian[2, 3] = gradient_angle

        return jacobian

    @staticmethod
    def kalman_update_formula(
        prior_mean,
        prior_covariance,
        measurement,
        predicted_measurement,
        measurement_matrix,
        measurement_noise_covariance,
    ):
        try:
            kalman_gain = (
                prior_covariance
                @ measurement_matrix.T
                @ np.linalg.inv(
                    measurement_matrix @ prior_covariance @ measurement_matrix.T
                    + measurement_noise_covariance
                )
            )
            updated_mean = prior_mean + kalman_gain @ (
                measurement - predicted_measurement
            )
            updated_covariance = (
                np.eye(prior_mean.size) - kalman_gain @ measurement_matrix
            ) @ prior_covariance
        except Exception as e:
            ros.logwarn(f"Unexpected exception: {e}")
            pyout(measurement)
            pyout(predicted_measurement)
        return updated_mean, updated_covariance

    @staticmethod
    def reset(service_name: str = "reset_kalman_filter"):
        try:
            # Create a service proxy
            reset_service = ros.ServiceProxy(service_name, Reset)
            # Call the service
            response = reset_service()
            # Log success if the service was called successfully
            ros.loginfo(
                f"Reset service '{service_name}' invoked successfully. Response: {response}"
            )
        except ros.ServiceException as e:
            # Log error if the service call failed
            ros.logerr(f"Service call failed: {e}")

    @staticmethod
    def pause():
        invoke_service("pause_kalman_filter")

    @staticmethod
    def unpause():
        invoke_service("unpause_kalman_filter")

    @staticmethod
    def get_state():
        try:
            service = ros.ServiceProxy("get_kalman_state", PODService)
            response = service()
            return pickle.loads(response.pod)
        except ros.ServiceException as e:
            ros.logerr(f"Service call failed: {e}")

    def __sensor_fusion(self, new_mean, new_covariance):
        self.mean, self.covariance = self.__add_new_keypoint_to_state(
            old_mean=self.mean,
            old_covariance=self.covariance,
            new_mean=new_mean,
            new_covariance=new_covariance,
        )

        modified_flag = True
        while modified_flag:
            try:
                for point_A_idx in range(0, self.mean.size, STATE_SIZE):
                    for point_B_idx in range(
                        point_A_idx + STATE_SIZE, self.mean.size, STATE_SIZE
                    ):
                        if self.__evaluate_keypoints_for_fusion(
                            point_A_idx=point_A_idx,
                            point_B_idx=point_B_idx,
                        ):
                            self.mean, self.covariance = self.__fuse_two_keypoints(
                                old_mean=self.mean,
                                old_covariance=self.covariance,
                                point_A_idx=point_A_idx,
                                point_B_idx=point_B_idx,
                                delete_duplicate=True,
                            )
                            raise BreakException
            except BreakException:
                modified_flag = True
            else:
                modified_flag = False

    def __add_new_keypoint_to_state(
        self,
        old_mean: np.ndarray,
        old_covariance: np.ndarray,
        new_mean: np.ndarray,
        new_covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate((old_mean, new_mean), axis=0)

        covariance = np.zeros(
            (
                old_covariance.shape[0] + new_covariance.shape[0],
                old_covariance.shape[1] + new_covariance.shape[1],
            )
        )
        covariance[: old_covariance.shape[0], : old_covariance.shape[1]] = (
            old_covariance
        )
        covariance[-new_covariance.shape[0] :, -new_covariance.shape[1] :] = (
            new_covariance
        )

        return mean, covariance

    def __evaluate_keypoints_for_fusion(
        self,
        point_A_idx: int,
        point_B_idx: int,
    ) -> bool:
        fused_mean, _ = self.__fuse_two_keypoints(
            old_mean=self.mean,
            old_covariance=self.covariance,
            point_A_idx=point_A_idx,
            point_B_idx=point_B_idx,
        )

        msk = np.mod(np.arange(self.covariance.shape[0]) + 1, 4) == 0
        msk = (msk[None, :] | msk[:, None]) * 1e4

        mahalanobis_distance = self.__calculate_mahalanobis_distance(
            point=fused_mean,
            mean=self.mean,
            covariance=self.covariance + msk,
        )
        if mahalanobis_distance > self.config["mahalanobis_distance_threshold"]:
            return False

        if not self.__is_within_camera_barrier(
            point=fused_mean[point_A_idx : point_A_idx + STATE_SIZE]
        ):
            return False

        return True

    def __fuse_two_keypoints(
        self,
        old_mean: np.ndarray,
        old_covariance: np.ndarray,
        point_A_idx: int,
        point_B_idx: int,
        eps: float = 1e-6,
        delete_duplicate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        distance_between_keypoints = np.zeros((STATE_SIZE, 1))
        predicted_distance_between_keypoints = (
            old_mean[point_A_idx : point_A_idx + STATE_SIZE]
            - old_mean[point_B_idx : point_B_idx + STATE_SIZE]
        )
        predicted_distance_between_keypoints[3] = (
            predicted_distance_between_keypoints[3] + np.pi
        ) % (2 * np.pi) - np.pi

        measurement_matrix = np.zeros((STATE_SIZE, old_mean.size))
        measurement_matrix[:, point_A_idx : point_A_idx + STATE_SIZE] = np.eye(
            STATE_SIZE
        )
        measurement_matrix[:, point_B_idx : point_B_idx + STATE_SIZE] = -np.eye(
            STATE_SIZE
        )

        measurement_noise_covariance = np.eye(STATE_SIZE) * eps

        new_mean, new_covariance = KalmanFilter.kalman_update_formula(
            prior_mean=old_mean,
            prior_covariance=old_covariance,
            measurement=distance_between_keypoints,
            predicted_measurement=predicted_distance_between_keypoints,
            measurement_matrix=measurement_matrix,
            measurement_noise_covariance=measurement_noise_covariance,
        )

        new_mean[point_A_idx + STATE_SIZE - 1] = (
            new_mean[point_A_idx + STATE_SIZE - 1] + np.pi
        ) % (2 * np.pi) - np.pi
        new_mean[point_B_idx + STATE_SIZE - 1] = (
            new_mean[point_B_idx + STATE_SIZE - 1] + np.pi
        ) % (2 * np.pi) - np.pi

        if delete_duplicate:
            new_mean, new_covariance = self.__remove_keypoint_from_distribution(
                old_mean=new_mean, old_covariance=new_covariance, point_idx=point_B_idx
            )

        return new_mean, new_covariance

    def __calculate_mahalanobis_distance(
        self, point: np.ndarray, mean: np.ndarray, covariance: np.ndarray
    ) -> float:
        nr_of_standard_deviations = np.sqrt(
            (point - mean).T @ np.linalg.inv(covariance) @ (point - mean)
        )
        return float(nr_of_standard_deviations)

    def __is_within_camera_barrier(self, point: np.ndarray) -> bool:
        point_xyz_world = point[:3]
        for camera_tcp in self.all_camera_tcps:
            camera_xyz_world = camera_tcp[:3, 3][:, None]
            camera_rotation_matrix = np.linalg.inv(camera_tcp[:3, :3])

            point_xyz_camera = camera_rotation_matrix @ (
                point_xyz_world - camera_xyz_world
            )

            point_z_camera = float(point_xyz_camera[2])

            if point_z_camera < self.config["min_camera_barrier_distance"]:
                return False
        return True

    def __remove_keypoint_from_distribution(
        self, old_mean: np.ndarray, old_covariance: np.ndarray, point_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        new_mean = np.concatenate(
            (old_mean[:point_idx], old_mean[point_idx + STATE_SIZE :]), axis=0
        )

        new_covariance = np.concatenate(
            (old_covariance[:point_idx], old_covariance[point_idx + STATE_SIZE :]),
            axis=0,
        ).copy()
        new_covariance = np.concatenate(
            (
                new_covariance[:, :point_idx],
                new_covariance[:, point_idx + STATE_SIZE :],
            ),
            axis=1,
        )

        return new_mean, new_covariance

    def __build_state_pod_message(
        self, timestamp: ros.Time, camera_tcp: Optional[np.ndarray]
    ) -> KalmanFilterStatePOD:
        means = self.mean.reshape((-1, STATE_SIZE))

        covariances = self.covariance.copy()
        for idx in range(STATE_SIZE, self.mean.size, STATE_SIZE):
            covariances[idx:] = np.roll(covariances[idx:], -STATE_SIZE, axis=1)
        covariances = covariances[:, :STATE_SIZE].reshape((-1, STATE_SIZE, STATE_SIZE))

        pod = KalmanFilterStatePOD(
            timestamp=timestamp,
            means=means,
            covariances=covariances,
            camera_tcp=camera_tcp,
        )

        return pod


def main():
    node = KalmanFilter()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
