import numpy as np


def homogenous_transformation(roll=0.0, pitch=0.0, yaw=0.0, dx=0.0, dy=0.0, dz=0.0):
    # Convert angles from degrees to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Rotation matrix around Z-axis
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Rotation matrix around Y-axis
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    # Rotation matrix around X-axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Homogeneous transformation matrix
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = np.array([dx, dy, dz])

    return H


def horizontal_view_rotation_matrix(z_axis: np.ndarray):
    x_axis = np.array([z_axis[1], -z_axis[0], 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.eye(4)
    R[:3, 0] = x_axis
    R[:3, 1] = y_axis
    R[:3, 2] = z_axis

    return R
