from pathlib import Path
from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
import rospkg
import pyrealsense2 as rs
import pyvista as pv
from pairo_butler.utils.tools import listdir, pbar, pyout


class POD3D:
    def __init__(self, image, depth, tcp):
        self.image: np.ndarray = image
        self.depth: np.ndarray = depth
        self.tcp: np.ndarray = tcp

        self.y: np.ndarray  # Measurement (N, 6)
        self.Q: np.ndarray  # Measurement noise covariance matrix (N, 6, 6)

        self.pixel_index: np.ndarray
        self.covariance_xyz: np.ndarray
        self.covariance_hsv: np.ndarray


class KalmanFilterState:
    def __init__(self, state_size=6):
        self.mu = np.empty((0, state_size))
        self.Sigma = np.empty((0, state_size, state_size))
        self.tcps = np.empty((0, 4, 4))

        self.mu_new: np.ndarray
        self.Sigma_new: np.ndarray


def initialize_camera_intrinsics():
    path = Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "rs2"
    try:
        depth_intrinsics_matrix = np.load(path / "depth.npy")
        color_intrinsics_matrix = np.load(path / "color.npy")
        extrinsics_matrix = np.load(path / "extrinsics.npy")
    except FileNotFoundError:
        # Define the resolution for both color and depth streams
        resolution_color = (640, 480)
        resolution_depth = (640, 480)
        fps = 30

        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Enable both color and depth streams
        config.enable_stream(rs.stream.depth, *resolution_depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, *resolution_color, rs.format.rgb8, 30)

        # Start the pipeline with the configuration
        pipeline.start(config)

        profiles = pipeline.get_active_profile()
        depth_profile = profiles.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profiles.get_stream(rs.stream.color).as_video_stream_profile()

        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        extrinsics = depth_profile.get_extrinsics_to(color_profile)

        depth_intrinsics_matrix = np.array(
            [
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1],
            ]
        )
        color_intrinsics_matrix = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ]
        )
        rotation = extrinsics.rotation
        translation = extrinsics.translation
        extrinsics_matrix = np.array(
            [
                [rotation[0], rotation[1], rotation[2], translation[0]],
                [rotation[3], rotation[4], rotation[5], translation[1]],
                [rotation[6], rotation[7], rotation[8], translation[2]],
                [0, 0, 0, 1],
            ]
        )

        np.save(path / "depth.npy", depth_intrinsics_matrix)
        np.save(path / "color.npy", color_intrinsics_matrix)
        np.save(path / "extrinsics.npy", extrinsics_matrix)

    return color_intrinsics_matrix, depth_intrinsics_matrix, extrinsics_matrix


def load_trial(path: Path):
    data_frames = []

    transform_path = (
        Path(rospkg.RosPack().get_path("airo_butler"))
        / "res"
        / "camera_tcps"
        / "T_rs2_sophie.npy"
    )
    transform_matrix = np.load(transform_path)
    for frame in pbar(listdir(path)):

        image = np.load(frame / "color.npy")[..., ::-1]
        depth = np.load(frame / "depth.npy")

        tcp_array = np.load(frame / "tcp.npy")
        tcp_camera = tcp_array @ transform_matrix

        data_frames.append(POD3D(image=image, depth=depth, tcp=tcp_camera))

    return data_frames


MIN_DEPTH = 200
MAX_DEPTH = 1000


def preprocess_measurements(trial: List[POD3D], intrinsics: np.ndarray) -> None:
    for frame in pbar(trial, desc="Preprocessing Measurements"):
        img = frame.image
        depth = frame.depth  # in millimeters
        tcp = frame.tcp  # in meters

        depth_mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)

        ix, iy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        K_inv = np.linalg.inv(intrinsics)

        pixels_homogeneous = np.stack([ix, iy, np.ones_like(ix)], axis=-1)

        pixel_index = np.arange(depth.size)[depth_mask.flatten()]
        pixels_homogeneous = pixels_homogeneous[depth_mask]
        depth = depth[depth_mask]
        bgr = img[depth_mask]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[depth_mask]

        cam_coords_homogeneous = (
            K_inv[None, ...] @ pixels_homogeneous[..., None]
        ).squeeze(-1) * depth[..., None]

        cam_coords = cam_coords_homogeneous / 1000  # convert to meters
        cam_coords_homogeneous = np.concatenate(
            (cam_coords, np.ones(cam_coords.shape[:-1] + (1,))), axis=-1
        )

        measured_points = tcp[None, ...] @ cam_coords_homogeneous[..., None]
        measured_points = measured_points.squeeze(-1)[..., :3]

        measurement = np.concatenate((measured_points, hsv), axis=1)
        frame.y = measurement
        frame.pixel_index = pixel_index

        # plot_pointcloud(measured_points, bgr)


def plot_pointcloud(points: np.ndarray, colors: np.ndarray) -> np.ndarray:

    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = (colors).astype(np.uint8)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(point_cloud, scalars="colors", rgb=True)
    labels = {"xlabel": "x", "ylabel": "y", "zlabel": "z"}
    plotter.add_axes(**labels)

    camera_position = [-5, 0, 0]
    focal_point = [-0.5, 0.0, 0.5]
    view_up = [0, 0, 1]
    plotter.camera_position = [camera_position, focal_point, view_up]

    plotter.show(auto_close=False)
    img_array = plotter.screenshot()
    plotter.close()

    cv2.imshow("points", img_array)
    cv2.waitKey(10)

    return img_array


ORTHOGONAL_MODIFIER = 0.5

ABSOLUTE_DEPTH_VARIANCE = 0.05
RELATIVE_DEPTH_VARIANCE = 0.15


def compute_covariance_over_position(trial: List[POD3D], intrinsics: np.ndarray):
    # Compute orthogonal variance matrix
    angles_between_pixels_x, angles_between_pixels_y = (
        initialize_orthogonal_angle_matrix(*trial[0].depth.shape, intrinsics)
    )
    angles_between_pixels_x = angles_between_pixels_x.reshape(-1)
    angles_between_pixels_y = angles_between_pixels_y.reshape(-1)

    for frame in pbar(trial, desc="Calculating Q Matrices"):
        depths = frame.depth.flatten()[frame.pixel_index] / 1000

        # compute orthogonal standard deviations
        angles_x = angles_between_pixels_x[frame.pixel_index] / 2
        angles_y = angles_between_pixels_y[frame.pixel_index] / 2

        pixel_widths = np.tan(angles_x) * depths
        pixel_height = np.tan(angles_y) * depths

        stdev_x = pixel_widths * ORTHOGONAL_MODIFIER
        stdev_y = pixel_height * ORTHOGONAL_MODIFIER

        # compute depth standard deviations
        stdev_z = ABSOLUTE_DEPTH_VARIANCE + RELATIVE_DEPTH_VARIANCE * depths

        # compute camera to point vectors
        points = frame.y[:, :3]
        vectors_z = points - frame.tcp[:3, 3][None, ...]
        vectors_z = vectors_z / np.linalg.norm(vectors_z, axis=-1, keepdims=True)
        vectors_x = np.stack(
            (vectors_z[:, 1], -vectors_z[:, 0], np.zeros_like(vectors_z[:, 0])), axis=-1
        )
        vectors_x = vectors_x / np.linalg.norm(vectors_x, axis=-1, keepdims=True)
        vectors_y = np.cross(vectors_z, vectors_x)

        R = np.stack((vectors_x, vectors_y, vectors_z), axis=-1)
        S = (
            np.eye(3)[None, ...]
            * np.stack([stdev_x**2, stdev_y**2, stdev_z**2], axis=-1)[..., None]
        )
        Q = R @ S @ R.transpose(0, 2, 1)

        frame.covariance_xyz = Q


def initialize_orthogonal_angle_matrix(height, width, M_intr):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    yx = np.stack((xx, yy), axis=-1)

    angles_y = np.full((height, width, 2), np.nan)
    angles_y[:-1, :, 0] = compute_angle_between_pixels(yx[:-1, :], yx[1:, :], M_intr)
    angles_y[1:, :, 1] = compute_angle_between_pixels(yx[1:, :], yx[:-1, :], M_intr)

    angles_x = np.full((height, width, 2), np.nan)
    angles_x[:, :-1, 0] = compute_angle_between_pixels(yx[:, :-1], yx[:, 1:], M_intr)
    angles_x[:, 1:, 1] = compute_angle_between_pixels(yx[:, 1:], yx[:, :-1], M_intr)

    angles_y = np.nanmean(angles_y, axis=-1)
    angles_x = np.nanmean(angles_x, axis=-1)

    angles_y[np.isnan(angles_y)] = np.nanmax(angles_y)
    angles_x[np.isnan(angles_x)] = np.nanmax(angles_x)

    return angles_x, angles_y


def pixel_to_normalized_camera_coords(pixel, intrinsics):
    """
    Convert pixel coordinates to normalized camera coordinates.
    """
    K_inv = np.linalg.inv(intrinsics)

    pixel_homogeneous = np.concatenate(
        (pixel, np.ones(pixel.shape[:-1] + (1,))), axis=-1
    )

    normalized_coords = K_inv[None, None] @ pixel_homogeneous[..., None]
    return normalized_coords.squeeze(-1)[..., :2]  # Return only x and y


def compute_angle_between_pixels(pixel1, pixel2, intrinsics):
    """
    Compute the angle between two pixels using the camera's intrinsics matrix.
    """
    # Convert pixels to normalized camera coordinates
    point1 = pixel_to_normalized_camera_coords(pixel1, intrinsics)
    point2 = pixel_to_normalized_camera_coords(pixel2, intrinsics)

    # Construct direction vectors in 3D space
    vector1 = np.concatenate((point1, np.ones((*point1.shape[:-1], 1))), axis=-1)
    vector2 = np.concatenate((point2, np.ones((*point2.shape[:-1], 1))), axis=-1)

    # Normalize the vectors
    vector1_norm = vector1 / np.linalg.norm(vector1, axis=-1)[..., None]
    vector2_norm = vector2 / np.linalg.norm(vector2, axis=-1)[..., None]

    # Compute the angle using the dot product
    cos_angle = (vector1_norm[..., None, :] @ vector2_norm[..., None]).reshape(
        point1.shape[:-1]
    )
    angle = np.arccos(cos_angle)  # Angle in radians

    return angle


BASE_STDEV_HUE = 5.0
MAX_STDEV_HUE = 180.0
BASE_STDEV_SATURATION = 50.0
BASE_STDEV_VALUE = 50.0


def compute_covariance_over_color(trial: List[POD3D]):
    for frame in trial:
        saturation = frame.y[:, 4]
        value = frame.y[:, 5]

        chroma = (saturation / 255) * (value / 255)

        hue_stdev = chroma * BASE_STDEV_HUE + (1 - chroma) * MAX_STDEV_HUE

        covariance = (
            np.eye(3)[None, ...]
            * np.stack(
                (
                    hue_stdev**2,
                    np.full_like(hue_stdev, BASE_STDEV_SATURATION) ** 2,
                    np.full_like(hue_stdev, BASE_STDEV_VALUE) ** 2,
                ),
                axis=-1,
            )[..., None]
        )

        frame.covariance_hsv = covariance


def construct_full_covariance_matrix(trial: List[POD3D]):
    for frame in trial:
        Q_xyz = frame.covariance_xyz
        Q_hsv = frame.covariance_hsv

        Q = np.zeros(
            (
                Q_xyz.shape[0],
                Q_xyz.shape[1] + Q_hsv.shape[1],
                Q_xyz.shape[1] + Q_hsv.shape[1],
            )
        )
        Q[:, : Q_xyz.shape[1], : Q_xyz.shape[1]] = Q_xyz
        Q[:, -Q_hsv.shape[1] :, -Q_hsv.shape[1] :] = Q_hsv

        frame.Q = Q


def add_new_measurements_to_state(frame: POD3D, state: KalmanFilterState):
    state.tcps = np.concatenate((state.tcps, frame.tcp[None, ...]), axis=0)
    state.mu_new = frame.y
    state.Sigma_new = frame.Q


FUSION_BATCH_SIZE = 10
SEED = 49


def landmark_fusion(state: KalmanFilterState):
    if state.mu.size == 0:
        return

    chunks = np.array_split(
        np.arange(state.mu.shape[0]), state.mu.shape[0] // FUSION_BATCH_SIZE + 1
    )

    changed = True
    while changed:
        changed = False
        for indexes in chunks:
            mu_old = state.mu[indexes]
            Sigma_old = state.Sigma[indexes]

            N_new = state.mu_new.shape[0]
            N_old = mu_old.shape[0]
            ss = mu_old.shape[1]

            M_mu = np.empty((N_new, N_old, ss * 2))
            M_Si = np.empty((N_new, N_old, ss * 2, ss * 2))

            M_mu[..., :ss] = np.tile(state.mu_new[:, None, :], (1, N_old, 1))
            M_mu[..., -ss:] = np.tile(mu_old[None, ...], (N_new, 1, 1))

            M_Si[..., :ss, :ss] = np.tile(
                state.Sigma_new[:, None, :, :], (1, N_old, 1, 1)
            )

            pyout()


def add_remaining_points_as_new_points(state: KalmanFilterState):
    state.mu = np.concatenate((state.mu, state.mu_new), axis=0)
    state.Sigma = np.concatenate((state.Sigma, state.Sigma_new), axis=0)
    state.mu_new = None
    state.Sigma_new = None
