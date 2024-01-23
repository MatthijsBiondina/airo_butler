# Arm poses
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import numpy as np
import rospkg
from pairo_butler.camera.rs2_camera import RS2Client
from pairo_butler.utils.point_cloud_utils import transform_points_to_different_frame
from pairo_butler.utils.tools import pyout
from pairo_butler.ur3_arms.ur3_client import UR3Client
from pairo_butler.camera.zed_camera import ZEDClient
import rospy as ros

SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi

SOPHIE_PULL = np.array([-0.3, 0.0, 0.1])
SOPHIE_RAISED = (np.array([-0.4, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]))

SOPHIE_CLOCK = np.array([-0.50, -1.0, +0.00, +0.00, -0.35, +0.00]) * np.pi
SOPHIE_MIDDLE = np.array([+0.00, -1.0, +0.50, -0.50, -0.50, +0.00]) * np.pi
SOPHIE_COUNTER = np.array([+0.60, -1.00, +0.25, -0.25, -0.75, +0.00]) * np.pi


WILSON_PREPARE_GRASP_STEP_1 = (
    np.array([+0.50, -0.00, -0.50, -0.50, -0.50, +0.00]) * np.pi
)
WILSON_PREPARE_GRASP_STEP_2 = (
    np.array([+0.50, -0.75, -0.75, -0.00, -0.50, +0.00]) * np.pi
)
WILSON_RAISED1 = np.array([+0.00, -0.60, -0.00, -0.90, +0.50, +0.00]) * np.pi
WILSON_RAISED2 = np.array([+0.00, -0.60, -0.00, -0.90, +1.50, +0.00]) * np.pi
WILSON_RAISED3 = np.array([+0.00, -0.60, -0.00, -0.90, -0.50, +0.00]) * np.pi
WILSON_RAISED4 = np.array([+0.00, -0.55, -0.00, -0.45, +0.50, +0.00]) * np.pi

# STATES
STATE_STARTUP = 0
STATE_PICKUP = 1
STATE_RAISE = 2
STATE_GRASP_CORNER = 3
STATE_SOPHIE_LET_GO = 4
STATE_SCAN = 5
STATE_RESET = 6
STATE_DONE = 7

# OTHER PARAMETERS
# the radius of the cylinder around the xy coordinate of Sophie's tcp when determining
# grasp point for Wilson. Too small and some of the towel will be cropped. Too large,
# and we will see sophie's arm.
SPREAD_FOR_WILSON_GRASP = 0.15


class PickUpTowelProcedure:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "pick_up_towel_procedure"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        self.rs2: Optional[RS2Client] = None
        self.zed: Optional[ZEDClient] = None
        self.wilson: Optional[UR3Client] = None
        self.sophie: Optional[UR3Client] = None

        # Placeholders
        self.state: str = STATE_STARTUP
        # self.state: str = STATE_SOPHIE_LET_GO
        (
            self.transformation_matrix_sophie_zed,
            self.transformation_matrix_wilson_zed,
        ) = self.__load_transformation_matrices()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        self.rs2 = RS2Client()
        self.zed = ZEDClient()
        self.wilson = UR3Client("wilson")
        self.sophie = UR3Client("sophie")

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.state == STATE_STARTUP:
                self.state = self.__startup()
            elif self.state == STATE_PICKUP:
                self.state = self.__pickup_towel()
            elif self.state == STATE_RAISE:
                self.state = self.__raise_towel()
            elif self.state == STATE_GRASP_CORNER:
                self.state = self.__grasp_corner()
            elif self.state == STATE_SOPHIE_LET_GO:
                self.state = self.__sophie_let_go()
            elif self.state == STATE_SCAN:
                self.state = self.__scan_towel()
            elif self.state == STATE_RESET:
                self.state = self.__reset_towel()
            elif self.state == STATE_DONE:
                ros.signal_shutdown("Done!")
                break
            else:
                ros.logwarn(f"Unknown state {self.state}.")
                raise ValueError(f"Unknown state {self.state}.")
            self.rate.sleep()

    def __startup(self):
        self.rs2.reset()

        self.sophie.open_gripper()
        self.wilson.open_gripper()
        self.sophie.move_to_joint_configuration(SOPHIE_REST)
        self.wilson.move_to_joint_configuration(WILSON_REST)

        return STATE_PICKUP

    def __compute_pickup(self):
        cloud = self.zed.pod.point_cloud
        xyz_in_zed_frame = cloud[:, :3]
        xyz_in_sophie_frame = transform_points_to_different_frame(
            xyz_in_zed_frame, self.transformation_matrix_sophie_zed
        )

        bounding_box_mask = (
            (xyz_in_sophie_frame[:, 0] < -0.2)
            & (xyz_in_sophie_frame[:, 0] > -0.7)
            & (xyz_in_sophie_frame[:, 1] < 0.5)
            & (xyz_in_sophie_frame[:, 1] > -0.5)
            & (xyz_in_sophie_frame[:, 2] > 0.01)
            & (xyz_in_sophie_frame[:, 2] < 0.2)
            & (np.linalg.norm(xyz_in_sophie_frame[:, :2], axis=1) > 0.3)
        )
        points = xyz_in_sophie_frame[bounding_box_mask]

        return points[np.argmax(points[:, 2])]

    def __pickup_towel(self):
        highest_point_on_towel = self.__compute_pickup()
        approach_point = np.copy(highest_point_on_towel)
        approach_point[2] += 0.1

        grasp_point = np.copy(highest_point_on_towel)
        grasp_point[2] -= 0.05
        grasp_point[2] = max(grasp_point[2], 0.02)

        self.sophie.move_to_tcp_vertical_down(approach_point)
        self.sophie.move_to_tcp_vertical_down(grasp_point)
        self.sophie.close_gripper()

        return STATE_RAISE

    def __compute_grasp(self) -> np.ndarray:
        timestamp = None
        accumulated_lowest_point_in_wilson_frame = None

        for _ in range(5):
            while timestamp is not None and self.zed.pod.timestamp == timestamp:
                self.rate.sleep()

            cloud = self.zed.pod.point_cloud
            points_in_zed_frame = cloud[:, :3]
            points_in_sophie_frame = transform_points_to_different_frame(
                points_in_zed_frame, self.transformation_matrix_sophie_zed
            )
            points_in_wilson_frame = transform_points_to_different_frame(
                points_in_zed_frame, self.transformation_matrix_wilson_zed
            )

            sophie_tcp = SOPHIE_RAISED[0]
            bounding_box_mask = (
                (points_in_sophie_frame[:, 2] > 0.35)
                & (points_in_sophie_frame[:, 2] < sophie_tcp[2])
                & (
                    np.sqrt(
                        (points_in_sophie_frame[:, 0] - sophie_tcp[0]) ** 2
                        + (points_in_sophie_frame[:, 1] - sophie_tcp[1]) ** 2
                    )
                    < SPREAD_FOR_WILSON_GRASP
                )
            )
            valid_points_in_wilson_frame = points_in_wilson_frame[bounding_box_mask]
            lowest_point_in_wilson_frame = valid_points_in_wilson_frame[
                np.argmin(valid_points_in_wilson_frame[:, 2])
            ]
            if (
                accumulated_lowest_point_in_wilson_frame is None
                or lowest_point_in_wilson_frame[1]
                < accumulated_lowest_point_in_wilson_frame[1]
            ):
                accumulated_lowest_point_in_wilson_frame = lowest_point_in_wilson_frame

        return accumulated_lowest_point_in_wilson_frame

    def __grasp_corner(self):
        self.wilson.move_to_joint_configuration(WILSON_PREPARE_GRASP_STEP_1)
        self.wilson.move_to_joint_configuration(WILSON_PREPARE_GRASP_STEP_2)
        lowest_point_on_towel = self.__compute_grasp()
        approach_point = lowest_point_on_towel - np.array([0.0, 0.0, 0.1])
        approach_point[2] = max(0.35, approach_point[2])
        grasp_point = lowest_point_on_towel
        grasp_point[2] += 0.02

        self.wilson.move_to_tcp_vertical_up(approach_point)
        self.wilson.move_to_tcp_vertical_up(grasp_point)
        self.wilson.close_gripper()

        return STATE_SOPHIE_LET_GO

    def __sophie_let_go(self):
        wilson_tcp = self.wilson.get_tcp_pose()
        z_diff = self.sophie.get_tcp_pose()[2, 3] - self.wilson.get_tcp_pose()[2, 3]

        self.wilson.move_to_tcp_vertical_up(
            np.array(
                [
                    wilson_tcp[0, 3],
                    SOPHIE_RAISED[0][1] + max(0.5 * z_diff, 0.2),
                    SOPHIE_RAISED[0][2],
                ]
            )
        )

        self.sophie.open_gripper()
        self.sophie.move_to_joint_configuration(SOPHIE_REST, blocking=False)

        self.wilson.move_to_joint_configuration(WILSON_RAISED1)
        self.wilson.move_to_joint_configuration(WILSON_RAISED4)

        self.sophie.move_to_joint_configuration(SOPHIE_CLOCK)

        ros.sleep(5)

        return STATE_SCAN

    def __scan_towel(self):
        self.sophie.move_to_joint_configuration(SOPHIE_CLOCK, joint_speed=0.1)
        self.sophie.move_to_joint_configuration(SOPHIE_MIDDLE, joint_speed=0.1)
        self.sophie.move_to_joint_configuration(SOPHIE_COUNTER, joint_speed=0.1)

        return STATE_RESET

    def __reset_towel(self):
        self.sophie.move_to_joint_configuration(SOPHIE_REST)
        self.wilson.move_to_joint_configuration(WILSON_RAISED1)

        x = 0.45
        y = np.random.uniform(-0.35, 0.35)
        z = 0.50

        self.wilson.move_to_tcp_vertical_down(np.array([x, y, z]))
        self.wilson.open_gripper()

        return STATE_STARTUP

    def __raise_towel(self):
        self.sophie.move_to_tcp_vertical_down(SOPHIE_PULL)
        self.sophie.move_to_tcp_horizontal(*SOPHIE_RAISED, blocking=False)
        self.wilson.move_to_joint_configuration(WILSON_PREPARE_GRASP_STEP_1)

        return STATE_GRASP_CORNER

    def __load_transformation_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        tcp_folder = (
            Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "camera_tcps"
        )

        transformation_matrix_zed_sophie = np.load(tcp_folder / "T_zed_sophie.npy")
        transformation_matrix_zed_wilson = np.load(tcp_folder / "T_zed_wilson.npy")

        return transformation_matrix_zed_sophie, transformation_matrix_zed_wilson


def main():
    node = PickUpTowelProcedure()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
