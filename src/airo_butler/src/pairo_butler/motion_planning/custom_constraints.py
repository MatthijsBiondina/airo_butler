from typing import Any, Callable
import numpy as np
from airo_typing import HomogeneousMatrixType
from cloth_tools.drake.scenes import X_CB_B, X_W_L_DEFAULT, X_W_R_DEFAULT
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.utils.tools import pyout
from ur_analytic_ik import ur5e


class DistanceBetweenToolsConstraint:
    def __init__(
        self,
        collision_check: Callable[..., bool],
        min_distance: float,
        max_distance: float,
        tcp_transform,
        
    ):
        self.collision_check = collision_check
        self.max_distance: float = max_distance
        self.min_distance: float = min_distance
        self.tcp_transform = np.array(tcp_transform)

        self.X_W_CB_wilson = (X_W_L_DEFAULT @ X_CB_B.inverse()).GetAsMatrix4()
        self.X_W_CB_sophie = (X_W_R_DEFAULT @ X_CB_B.inverse()).GetAsMatrix4()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        collision_ok = self.collision_check(*args, **kwds)

        joints_wilson = args[0][0:6]

        X_CB_TCP_wilson = ur5e.forward_kinematics_with_tcp(
            *list(joints_wilson), self.tcp_transform
        )
        X_W_TCP_wilson = self.X_W_CB_wilson @ X_CB_TCP_wilson

        joints_sophie = args[0][6:12]
        X_CB_TCP_sophie = ur5e.forward_kinematics_with_tcp(
            *list(joints_sophie), self.tcp_transform
        )
        X_W_TCP_sophie = self.X_W_CB_sophie @ X_CB_TCP_sophie

        distance = np.linalg.norm(X_W_TCP_wilson[:3, 3] - X_W_TCP_sophie[:3, 3])

        if (
            collision_ok
            and distance < self.max_distance
            and distance > self.min_distance
        ):
            return True
        else:
            return False
