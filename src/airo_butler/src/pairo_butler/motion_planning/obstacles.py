from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_CHARUCO_BOARD,
)
import airo_models


class CharucoBoard:

    def __init__(self):
        self.width = (
            AIRO_DEFAULT_CHARUCO_BOARD.getChessboardSize()[0]
            * AIRO_DEFAULT_CHARUCO_BOARD.getSquareLength()
        )
        self.height = (
            AIRO_DEFAULT_CHARUCO_BOARD.getChessboardSize()[1]
            * AIRO_DEFAULT_CHARUCO_BOARD.getSquareLength()
        )
        self.thickness = 0.01
        self.square_size = AIRO_DEFAULT_CHARUCO_BOARD.getSquareLength()

        self.urdf = airo_models.box_urdf_path(
            (self.width, self.height, self.thickness), "charuco_board"
        )


class HangingTowel:

    def __init__(self):
        self.radius = 0.1
        self.bottom = 0.01
        self.top = 0.60

        self.length = self.top - self.bottom
        self.urdf = airo_models.box_urdf_path(
            (self.radius * 2, self.radius * 2, self.length), "towel"
        )
