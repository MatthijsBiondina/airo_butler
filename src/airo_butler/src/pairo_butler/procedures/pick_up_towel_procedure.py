# Arm poses
from pathlib import Path
import time
from typing import Optional
import numpy as np
import rospkg
from pairo_butler.utils.point_cloud_utils import transform_points_to_different_frame
from pairo_butler.utils.tools import pyout
from pairo_butler.ur3_arms.ur3_client import UR3Client
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros

SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi


# STATES
STATE_STARTUP = 0
STATE_COMPUTE_PICKUP = 1
STATE_PICKUP = 2
STATE_RAISE = 3
STATE_DONE = 4


class PickUpTowelProcedure:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "pick_up_towel_procedure"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        self.zed: Optional[ZEDClient] = None
        self.wilson: Optional[UR3Client] = None
        self.sophie: Optional[UR3Client] = None

        # Placeholders
        self.state: str = STATE_STARTUP
        self.tranformation_matrix_sophie_zed: np.ndarray = (
            self.__load_transformation_matrix()
        )
        self.highest_point_on_towel: Optional[np.ndarray] = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.zed = ZEDClient()
        self.wilson = UR3Client("wilson")
        self.sophie = UR3Client("sophie")

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.state == STATE_STARTUP:
                self.state = self.__startup()
            elif self.state == STATE_COMPUTE_PICKUP:
                self.state = self.__compute_pickup()
            elif self.state == STATE_PICKUP:
                self.state = self.__pickup_towel()
            elif self.state == STATE_RAISE:
                self.state = self.__raise_towel()
            elif self.state == STATE_DONE:
                ros.signal_shutdown("Done!")
                break
            else:
                ros.logwarn(f"Unknown state {self.state}.")
                raise ValueError(f"Unknown state {self.state}.")
            self.rate.sleep()

    def __startup(self):
        ros.sleep(3)
        self.sophie.open_gripper()
        self.wilson.open_gripper()
        ros.sleep(3)
        self.sophie.move_to_joint_configuration(SOPHIE_REST)
        self.wilson.move_to_joint_configuration(WILSON_REST)

        return STATE_COMPUTE_PICKUP

    def __compute_pickup(self):
        cloud = self.zed.pod.point_cloud
        xyz_in_zed_frame = cloud[:, :3]
        xyz_in_sophie_frame = transform_points_to_different_frame(
            xyz_in_zed_frame, self.tranformation_matrix_sophie_zed
        )

        bounding_box_mask = (
            (xyz_in_sophie_frame[:, 0] < -0.2)
            & (xyz_in_sophie_frame[:, 0] > -0.7)
            & (xyz_in_sophie_frame[:, 1] < 0.5)
            & (xyz_in_sophie_frame[:, 1] > -0.5)
            & (xyz_in_sophie_frame[:, 2] > 0.01)
            & (xyz_in_sophie_frame[:, 2] < 0.2)
        )
        points = xyz_in_sophie_frame[bounding_box_mask]

        self.highest_point_on_towel = points[np.argmax(points[:, 2])]

        return STATE_PICKUP

    def __pickup_towel(self):
        approach_point = np.copy(self.highest_point_on_towel)
        approach_point[2] += 0.1

        grasp_point = np.copy(self.highest_point_on_towel)
        grasp_point[2] -= 0.02
        grasp_point[2] = max(grasp_point[2], 0.02)

        self.sophie.move_to_tcp_vertical_down(approach_point)
        self.sophie.move_to_tcp_vertical_down(grasp_point)
        self.sophie.close_gripper()

        return STATE_RAISE

    def __raise_towel(self):
        self.sophie.move_to_tcp_horizontal(
            np.array([-0.4, 0.0, 1.0]), z=np.array([-1.0, 0.0, 0.0])
        )
        return STATE_DONE

    def __load_transformation_matrix(self) -> np.ndarray:
        fpath = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "res"
            / "camera_tcps"
            / "T_zed_sophie.npy"
        )

        transformation_matrix = np.load(fpath)
        return transformation_matrix


def main():
    node = PickUpTowelProcedure()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
