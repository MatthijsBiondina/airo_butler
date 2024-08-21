from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F
from pairo_butler.SplaTAM.viz_scripts.final_recon import (
    load_camera,
    load_scene_data,
    render,
    rgbd2pcd,
)
from pairo_butler.SplaTAM.utils.common_utils import seed_everything
from pairo_butler.SplaTAM.utils.slam_external import build_rotation
from pairo_butler.utils.tools import UGENT, pyout
from scipy.spatial.transform import Rotation as R
import open3d as o3d

splatam_root = Path(
    "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/experiments/TUM/towel_splatam"
)

# Load the .npz file
params = np.load(splatam_root / "params.npz")


# compute splatam camera poses
def make_homogeneous_matrix(translation, quaternion):
    # Convert the quaternion to a rotation matrix
    rotation = R.from_quat(quaternion)
    R_matrix = rotation.as_matrix()  # 3x3 rotation matrix

    # Create a 4x4 identity matrix
    T = np.eye(4)

    # Assign the rotation matrix to the top-left 3x3 part of the transformation matrix
    T[:3, :3] = R_matrix

    # Assign the translation vector to the top-right 3x1 part of the transformation matrix
    T[:3, 3] = translation

    return T


def matrix_to_translation_quaternion(T):
    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    return (*translation, *quaternion)


def load_splatam_camera_pose_estimates():
    all_w2cs = []
    num_t = params["cam_unnorm_rots"].shape[-1]
    for t_i in range(num_t):
        cam_rot = params["cam_unnorm_rots"][..., t_i]
        cam_rot = cam_rot / np.linalg.norm(cam_rot)

        cam_tran = params["cam_trans"][..., t_i]

        rel_w2c = np.eye(4)

        rel_w2c[:3, :3] = np.array(build_rotation(Tensor(cam_rot)).cpu())
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(np.linalg.inv(rel_w2c))

    w2cs = np.stack(all_w2cs, axis=0)

    # M = np.array(
    #     [
    #         [-1.0, 0.0, 0.0, 0.0],
    #         [0.0, -1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    # w2cs = w2cs @ M[None, ...]

    return w2cs


def load_rs2_camera_poses_measured():
    file_path = "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS/rgbd_dataset_0/pose.txt"
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            lines.append(line)
    lines = lines[3:]
    gt_data = np.array([[float(d) for d in line.split(" ")] for line in lines])

    gt_tcps = np.stack(
        [make_homogeneous_matrix(pose[1:4], pose[4:]) for pose in gt_data], axis=0
    )

    return gt_tcps


def umeyama_alignment(source, target):
    """
    Perform the Umeyama alignment to find the best rigid transformation (rotation and translation)
    that aligns the source points to the target points.

    Args:
        source (np.ndarray): Nx3 matrix of source points (T_splatam_splatamworldframe).
        target (np.ndarray): Nx3 matrix of target points (T_rs2_rs2worldframe).

    Returns:
        T (np.ndarray): 4x4 transformation matrix that best aligns source to target.
    """
    assert source.shape == target.shape, "Source and target must have the same shape."

    # Compute centroids of the source and target points
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)

    # Center the points around the centroids
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute the covariance matrix
    covariance_matrix = np.dot(source_centered.T, target_centered) / source.shape[0]

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix
    R_matrix = np.dot(Vt.T, U.T)

    # Ensure the rotation matrix is a proper rotation (det(R) should be +1)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation = target_centroid - np.dot(R_matrix, source_centroid)

    # Construct the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = translation

    return T


def compute_T_sim_real(T_rs_sim, T_rs_real):
    # Extract translations from the 4x4 matrices
    sim_positions = np.concatenate(
        (
            T_rs_sim[:, :3, 3],
            T_rs_sim[:, :3, 3] + T_rs_sim[:, :3, 0],
            T_rs_sim[:, :3, 3] + T_rs_sim[:, :3, 1],
            T_rs_sim[:, :3, 3] + T_rs_sim[:, :3, 2],
        ),
        axis=0,
    )
    real_positions = np.concatenate(
        (
            T_rs_real[:, :3, 3],
            T_rs_real[:, :3, 3] + T_rs_real[:, :3, 0],
            T_rs_real[:, :3, 3] + T_rs_real[:, :3, 1],
            T_rs_real[:, :3, 3] + T_rs_real[:, :3, 2],
        ),
        axis=0,
    )

    # Find the best-fit transformation from splatamworldframe to rs2worldframe
    T_sim_real = umeyama_alignment(sim_positions, real_positions)

    return T_sim_real


def transform_points(points, T):
    # Convert points to homogeneous coordinates by adding a column of ones
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation
    transformed_points = (T @ points_homogeneous.T).T

    # Convert back to Cartesian coordinates
    return transformed_points[:, :3]


def hex_to_rgb(hex_color):
    # Remove the hash symbol (#) if it's present
    hex_color = hex_color.lstrip("#")

    # Convert the hex string to an integer tuple (R, G, B)
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    return rgb


def plot_camera_poses(camera_poses, color):
    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]),
        o3d.geometry.TriangleMesh.create_sphere(radius=0.01),
    ]
    geometries[-1].paint_uniform_color([c / 255 for c in color])

    for pose in camera_poses:
        R = pose[:3, :3]
        t = pose[:3, 3]

        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        camera_frame.transform(pose)

        # Add the camera position (translation) as a small sphere
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        camera_sphere.translate(t)
        camera_sphere.paint_uniform_color(
            [c / 255 for c in color]
        )  # Red color for visibility

        geometries.append(camera_frame)
        geometries.append(camera_sphere)

    # Create the Open3D visualization
    return geometries


def plot_points(means3D, rgb_color, distance_threshold=0.2, alpha_percentile=75):

    distance_mask = np.linalg.norm(means3D[:, :2], axis=1) < distance_threshold

    opacities = params["logit_opacities"]
    percentile = np.percentile(opacities, alpha_percentile)
    opacity_mask = (opacities > percentile).squeeze(-1)

    mask = distance_mask & opacity_mask

    filtered_points = means3D[mask]
    filtered_colors = params["rgb_colors"][mask]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return [point_cloud]


def save_masked_data(means3D, distance_threshold=0.2):
    mask = np.linalg.norm(means3D[:, :2], axis=1) < distance_threshold

    means3D_masked = means3D[mask]
    rgb_masked = params["rgb_colors"][mask]
    alpha_masked = params["logit_opacities"][mask]
    scales_masked = params["log_scales"][mask]

    np.savez(
        splatam_root / "pointcloud.npz",
        means3D=means3D_masked,
        rgb=rgb_masked,
        alpha_logits=alpha_masked,
        log_scales=scales_masked,
    )


if __name__ == "__main__":
    nskip = 5

    T_cam_sim_estimated = load_splatam_camera_pose_estimates()
    T_cam_real_measured = load_rs2_camera_poses_measured()

    T_sim_real = compute_T_sim_real(T_cam_sim_estimated, T_cam_real_measured)

    T_cam_real_estimated = T_sim_real @ T_cam_sim_estimated
    means3D_real = transform_points(params["means3D"], T_sim_real)

    save_masked_data(means3D_real)

    camera_estimated = plot_camera_poses(
        T_cam_real_estimated[::nskip], hex_to_rgb(UGENT.PINK)
    )
    camera_measured = plot_camera_poses(
        T_cam_real_measured[::nskip], hex_to_rgb(UGENT.BLUE)
    )
    points_real = plot_points(means3D_real, params["rgb_colors"])

    geometries = camera_estimated + camera_measured + points_real

    o3d.visualization.draw_geometries(geometries)
