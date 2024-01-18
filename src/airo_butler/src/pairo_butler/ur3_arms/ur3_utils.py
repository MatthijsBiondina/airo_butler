from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
import numpy as np

RotVecPoseType = np.ndarray
""" a 6D pose [tx,ty,tz,rotvecx,rotvecy,rotvecz]"""


def convert_homegeneous_pose_to_rotvec_pose(
    homogeneous_pose: HomogeneousMatrixType,
) -> RotVecPoseType:
    se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
    rotation = se3.orientation_as_rotation_vector
    translation = se3.translation
    return np.concatenate([translation, rotation])
