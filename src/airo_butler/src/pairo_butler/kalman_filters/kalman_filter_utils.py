from pathlib import Path
import time
from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
import rospkg
import pyrealsense2 as rs
import pyvista as pv
import torch
from torch import Tensor
from pairo_butler.utils.tools import listdir, pbar, pyout

torch.set_printoptions(precision=3, sci_mode=False)


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
    def __init__(self, state_size=6, gpu="cpu"):
        self.size = state_size
        self.device = torch.device(gpu)
        self.mu: Tensor = torch.empty((0, state_size), dtype=torch.float64).to(
            self.device
        )
        self.Sigma: Tensor = torch.empty(
            (0, state_size, state_size), dtype=torch.float64
        ).to(self.device)
        self.tcps: Tensor = torch.empty((0, 4, 4)).to(self.device)

        self.mu_new: Tensor
        self.Sigma_new: Tensor


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
STRIDE = 10


def preprocess_measurements(trial: List[POD3D], intrinsics: np.ndarray) -> None:
    for frame in pbar(trial, desc="Preprocessing Measurements"):
        img = frame.image
        depth = frame.depth  # in millimeters
        tcp = frame.tcp  # in meters

        depth_mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
        stride_mask = np.zeros_like(depth_mask)
        stride_mask[::STRIDE, ::STRIDE] = True
        mask = depth_mask & stride_mask

        pixel_index = np.arange(depth.size)[mask.flatten()]

        ix, iy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        K_inv = np.linalg.inv(intrinsics)

        pixels_homogeneous = np.stack([ix, iy, np.ones_like(ix)], axis=-1)
        pixels_homogeneous = pixels_homogeneous[mask]
        depth = depth[mask]
        bgr = img[mask]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[mask]

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

    camera_position = [0, -5, 0]
    focal_point = [-0.5, 0.0, 0.5]
    view_up = [0, 0, 1]
    plotter.camera_position = [camera_position, focal_point, view_up]

    plotter.show(auto_close=False)
    img_array = plotter.screenshot()
    plotter.close()

    cv2.imshow("points", img_array)
    cv2.waitKey(10)

    return img_array


def plot_state(state: KalmanFilterState, threshold=0.001):
    M = torch.linalg.det(state.Sigma) < threshold
    if not torch.any(M):
        return
    points = state.mu[M][:, :3].cpu().numpy()
    hsv = state.mu[M][:, 3:].cpu().numpy().astype(np.uint8)
    bgr = cv2.cvtColor(hsv[:, None, :], cv2.COLOR_HSV2BGR).squeeze(1)

    plot_pointcloud(points, bgr)


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

    return angles_x * STRIDE, angles_y * STRIDE


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

    state.tcps = torch.cat(
        (state.tcps, torch.tensor(frame.tcp[None, ...]).to(state.device))
    )
    state.mu_new = torch.tensor(frame.y, dtype=torch.float32).to(state.device)
    state.Sigma_new = torch.tensor(frame.Q, dtype=torch.float32).to(state.device)


FUSION_BATCH_SIZE = 100
MAHALANOBIS_THRESHOLD = 2.0


def landmark_fusion(state: KalmanFilterState):
    if state.mu.shape[0] == 0:
        return

    chunks = torch.tensor_split(
        torch.arange(state.mu.shape[0]).to(state.device),
        state.mu.shape[0] // FUSION_BATCH_SIZE + 1,
    )

    # Placeholders
    mu = torch.empty(
        state.mu_new.shape[0], FUSION_BATCH_SIZE, state.size * 2, dtype=torch.float64
    ).to(state.device)
    Sigma = torch.empty(
        state.mu_new.shape[0],
        FUSION_BATCH_SIZE,
        state.size * 2,
        state.size * 2,
        dtype=torch.float64,
    ).to(state.device)
    # Kalman filter
    C = torch.zeros((state.size, state.size * 2), dtype=torch.float64).to(state.device)
    C[:, : state.size] = torch.eye(state.size).to(state.device)
    C[:, state.size :] = -torch.eye(state.size).to(state.device)
    Q = torch.eye(state.size, dtype=torch.float64).to(state.device) * 1e-5
    eye = torch.eye(state.size * 2).to(state.device)[None, None, ...]

    changed = True
    while changed:
        changed = False
        for indexes in pbar(chunks):  # indexes corresponds with elements from state.mu
            mu_old = state.mu[indexes]
            Sigma_old = state.Sigma[indexes]

            N_new = state.mu_new.shape[0]
            N_old = mu_old.shape[0]
            state_size = mu_old.shape[1]

            if N_new == 0:
                break

            mu[:N_new, :N_old, :state_size] = torch.tile(
                state.mu_new[:, None, :], (1, N_old, 1)
            )
            mu[:N_new, :N_old, state_size:] = torch.tile(
                mu_old[None, ...], (N_new, 1, 1)
            )

            Sigma[:N_new, :N_old, :state_size, :state_size] = torch.tile(
                state.Sigma_new[:, None, :, :], (1, N_old, 1, 1)
            )
            Sigma[:N_new, :N_old, state_size:, state_size:] = torch.tile(
                Sigma_old[None, ...], (N_new, 1, 1, 1)
            )
            dim = torch.arange(12).to(Sigma.device)

            # Clip to prevent overflow
            Sigma[:, :, dim, dim][Sigma[:, :, dim, dim] < 1e-5] = 1e-5
            Sigma = torch.clamp(Sigma, -1e6, 1e6)

            mu[:N_new, :N_old] = hue_circular_wrap_around_handling(
                mu[:N_new, :N_old], state, N_new, N_old
            )
            mu_bar, Sigma_bar = kalman_formulas(
                mu[:N_new, :N_old], Sigma[:N_new, :N_old], C, Q, N_new, N_old, eye
            )

            d = mahalanobis_distance_matrix(
                mu[:N_new, :N_old], mu_bar, Sigma[:N_new, :N_old]
            )

            indexes_mu_bar_dim0, indexes_mu_bar_dim1, indexes_state_dim0 = (
                pair_merge_matches(d, indexes)
            )

            if indexes_mu_bar_dim0.shape[0] > 0:
                changed = True

                # Add updated to state
                state.mu[indexes_state_dim0] = mu_bar[
                    indexes_mu_bar_dim0, indexes_mu_bar_dim1, :state_size
                ]
                state.Sigma[indexes_state_dim0] = Sigma_bar[
                    indexes_mu_bar_dim0, indexes_mu_bar_dim1, :state_size, :state_size
                ]

                # Remove merged measurements from state new:
                remaining_new_indexes = torch.cat(
                    (
                        torch.arange(0, indexes_mu_bar_dim0[0]),
                        *[
                            torch.arange(
                                indexes_mu_bar_dim0[ii] + 1, indexes_mu_bar_dim0[ii + 1]
                            )
                            for ii in range(indexes_mu_bar_dim0.shape[0] - 1)
                        ],
                        torch.arange(indexes_mu_bar_dim0[-1] + 1, mu_bar.shape[0]),
                    ),
                    dim=0,
                )
                state.mu_new = state.mu_new[remaining_new_indexes]
                state.Sigma_new = state.Sigma_new[remaining_new_indexes]


def hue_circular_wrap_around_handling(
    mu: torch.Tensor, state: KalmanFilterState, N_new: int, N_old: int
) -> Tensor:
    # Hue is circular (circular wrap-around handling)
    HUE_IDX = 4
    hues_old = mu[:N_new, :N_old, HUE_IDX]
    hues_new = mu[:N_new, :N_old, state.size + HUE_IDX]
    hues_new = torch.stack((hues_new - 180, hues_new, hues_new + 180), dim=-1)

    hues_idx = torch.argmin(
        torch.abs(hues_old[..., None] - hues_new), dim=-1, keepdim=True
    )
    hues_new = torch.gather(hues_new, dim=-1, index=hues_idx).squeeze(-1)
    mu[:N_new, :N_old, state.size + HUE_IDX] = hues_new
    return mu


def kalman_formulas(mu, Sigma, C, Q, N_new, N_old, eye, eps=1e-4):
    dim = torch.arange(6).to(Sigma.device)
    inside_inverse = (
        C[None, None, ...] @ Sigma @ C.T[None, None, ...] + Q[None, None, ...]
    )
    inside_inverse[:, :, dim, dim][inside_inverse[:, :, dim, dim] < eps] = eps

    try:
        K = Sigma @ C.T[None, None, ...] @ torch.linalg.inv(inside_inverse)
    except Exception as e:
        el = inside_inverse.reshape(-1, 6, 6)
        idx = int(str(e).split(" ")[3].replace("):", ""))
        pyout(el[idx])

    mu_bar = (
        mu[:N_new, :N_old, :, None]
        + K @ -C[None, None, ...] @ mu[:N_new, :N_old, :, None]
    ).squeeze(-1)
    Sigma_bar = (eye - K @ C[None, None, ...]) @ Sigma

    return mu_bar, Sigma_bar


def mahalanobis_distance_matrix(mu1: Tensor, mu2: Tensor, Sigma: Tensor) -> Tensor:
    diff = (mu1 - mu2)[..., None]

    try:
        ininv = Sigma + torch.eye(12)[None, None, ...] * 1e-6
        ininv = torch.clamp(ininv, -1e4, 1e4)
        D = (
            (diff.transpose(-1, -2) @ torch.linalg.inv(ininv) @ diff)
            .squeeze(-1)
            .squeeze(-1)
        )
    except Exception as e:
        el = ininv.reshape(-1, 12, 12)
        idx = int(str(e).split(" ")[3].replace("):", ""))
        pyout(el[idx])
        pyout(e)

    return D


def pair_merge_matches(d: torch.Tensor, indexes_state: torch.Tensor) -> torch.Tensor:
    if indexes_state.shape[0] == 0:
        return (
            torch.LongTensor([]).to(d.device),
            torch.LongTensor([]).to(d.device),
            torch.LongTensor([]).to(d.device),
        )

    # Define the pair_merge_matches function that merges landmark pairs based on distance criteria

    # Create a tensor of sequential integers with the same length as the second dimension of `d`
    # This tensor represents column indices for `d`
    dim0 = torch.arange(d.shape[0]).to(d.device)  # dim0 of d
    dim1 = torch.arange(d.shape[1]).to(d.device)  # dim1 of d

    # Find the index of the minimum value in `d` across rows for each column, indicating the closest landmark
    indexes_dim0 = torch.argmin(d, dim=0)
    indexes_dim1 = dim1

    # Extract the distances of the closest landmarks
    distances_closest = d[indexes_dim0, indexes_dim1]

    # Create a mask to filter out pairs with a distance above a predefined threshold
    distance_mask = distances_closest < MAHALANOBIS_THRESHOLD

    # Apply the mask to filter out non-matching landmarks based on the distance threshold
    indexes_dim0 = indexes_dim0[distance_mask]
    indexes_dim1 = indexes_dim1[distance_mask]
    indexes_state = indexes_state[distance_mask]

    # Sort the matching pairs by the new indexes to ensure order consistency
    sorted_order = torch.argsort(indexes_dim0)

    # Reorder the matched indexes based on the sorted order
    indexes_dim0 = indexes_dim0[sorted_order]
    indexes_dim1 = indexes_dim1[sorted_order]
    indexes_state = indexes_state[sorted_order]

    # Create a mask to remove duplicate entries in matched_indexes_new to ensure unique matching
    if indexes_dim0.shape[0] == 0:
        duplicate_mask = torch.BoolTensor([]).to(indexes_dim0.device)
    else:
        duplicate_mask = torch.cat(
            (
                torch.BoolTensor([True]).to(
                    indexes_dim0.device
                ),  # Always keep the first element
                indexes_dim0[1:]
                != indexes_dim0[:-1],  # Keep non-duplicate subsequent entries
            ),
            dim=0,
        )
    # Apply the mask to both sets of indexes to finalize the unique pairings
    indexes_dim0 = indexes_dim0[duplicate_mask]
    indexes_dim1 = indexes_dim1[duplicate_mask]
    indexes_state = indexes_state[duplicate_mask]

    # Return the final matched indexes of new and old landmarks
    return indexes_dim0, indexes_dim1, indexes_state


def add_remaining_points_as_new_points(state: KalmanFilterState):
    state.mu = torch.cat((state.mu, state.mu_new), dim=0)
    state.Sigma = torch.cat((state.Sigma, state.Sigma_new), dim=0)
    state.mu_new = None
    state.Sigma_new = None


def scale_hue_within_range(state: KalmanFilterState):
    state.mu[:, 4] = torch.remainder(state.mu[:, 4], 180)
