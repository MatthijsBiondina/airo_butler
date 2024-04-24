import cv2
from cv2 import aruco
import numpy as np

from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_aruco_markers,
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
)

root = "/home/matt/Pictures"
img = cv2.imread(f"{root}/frame.png")
intrinsics = np.load(f"{root}/intrinsics.npy")
aruco_dict = AIRO_DEFAULT_ARUCO_DICT
charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

aruco_result = detect_aruco_markers(img, aruco_dict)


print(f"Version: {cv2.__version__}")

print("Foo!")
nb_corners, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
    markerCorners=aruco_result.corners,  # type: ignore # typed as Seq but accepts np.ndarray
    markerIds=aruco_result.ids,
    image=img,
    board=charuco_board,
)
print("Bar!")
