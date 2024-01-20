import cv2
import numpy as np

from pairo_butler.utils.tools import pyout


def transform_points_to_different_frame(
    xyz_in_zed_frame: np.ndarray, transformation_matrix: np.ndarray
) -> np.ndarray:
    """
    Transforms a batch of points from the ZED camera frame to a different frame
    using a transformation matrix.

    Args:
        xyz_in_zed_frame (np.ndarray): The Nx3 numpy array of points in the ZED
        camera frame.
        transformation_matrix (np.ndarray): The 4x4 transformation matrix used to
        transform the points.

    Returns:
        np.ndarray: The transformed Nx3 numpy array of points in the new frame.
    """
    # Convert the XYZ coordinates to homogeneous coordinates
    points_homogenous = np.hstack(
        (xyz_in_zed_frame, np.ones((xyz_in_zed_frame.shape[0], 1)))
    )

    transformation_matrix = transformation_matrix[None, ...]
    points_homogenous = points_homogenous[..., None]

    # Apply the transformation matrix to the homogeneous coordinates
    transformed_points_homegenous = transformation_matrix @ points_homogenous
    transformed_points_homegenous = transformed_points_homegenous.squeeze(-1)

    # Convert back from homogeneous coordinates to 3D coordinates

    transformed_points = (
        transformed_points_homegenous[:, :3]
        / transformed_points_homegenous[:, -1][:, None]
    )

    return transformed_points


def rgb_to_hue(rgb_array):
    # Convert the RGB array to the expected shape (1, N, 3) for OpenCV
    rgb_array_reshaped = rgb_array.reshape(1, -1, 3)

    # Convert RGB to HSV
    hsv_array = cv2.cvtColor(rgb_array_reshaped.astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Extract the Hue component and reshape back to (N, 1)
    hue_array = hsv_array[0, :, 0].reshape(-1, 1)

    return hue_array


# Example usage
rgb_array = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
)  # Replace with your Nx3 RGB array
hue_array = rgb_to_hue(rgb_array)
