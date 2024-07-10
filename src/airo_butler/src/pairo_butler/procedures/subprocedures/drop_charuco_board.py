import numpy as np
from pairo_butler.utils.tools import pyout
from pairo_butler.motion_planning.obstacles import CharucoBoard
from pairo_butler.utils.transformations_3d import homogenous_transformation
from pairo_butler.camera.calibration import CameraCalibration, TCPs
from pairo_butler.procedures.subprocedure import Subprocedure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rospkg
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros


class DropCharucoBoard(Subprocedure):
    SCENE = "sophie_holds_charuco"
    N_MEASUREMENTS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.board = CharucoBoard()

    def run(self):
        tcp_drop = np.array(
            [
                [0.0, 0.0, -1.0, 0.05],
                [0.0, -1.0, 0.0, self.board.height / 2],
                [-1.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        plan = self.ompl.plan_to_tcp_pose(sophie=tcp_drop, scene="sophie_holds_charuco")
        self.sophie.execute_plan(plan)

        self.sophie.open_gripper()

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie), scene="default"
        )
        self.sophie.execute_plan(plan)
