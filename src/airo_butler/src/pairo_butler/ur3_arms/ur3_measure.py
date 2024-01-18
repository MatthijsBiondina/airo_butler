import time
from typing import Optional
import numpy as np
from pairo_butler.utils.tools import pyout
from pairo_butler.ur3_arms.ur3_client import UR3Client
import rospy as ros


SOPHIE_REST = np.array([+0.00, -1.00, +0.50, -0.50, -0.50, +0.00]) * np.pi
WILSON_REST = np.array([+0.00, -0.00, -0.50, -0.50, +0.50, +0.00]) * np.pi


STATE_WARMUP = 0
STATE_MEASURE = 1
STATE_TEST = 2
STATE_DONE = 3

np.set_printoptions(suppress=True, precision=3)


class UR3Measurer:
    PUBLISH_RATE = 30
    QUEUE_SIZE = 2

    def __init__(self, name: str = "ur3_measurer") -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.wilson: Optional[UR3Client] = None
        self.sophie: Optional[UR3Client] = None

        self.state: int = STATE_WARMUP

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.wilson = UR3Client("left")
        self.sophie = UR3Client("right")
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if self.state == STATE_WARMUP:
                self.state = self.__startup()
            elif self.state == STATE_MEASURE:
                self.state = self.__measure()
            elif self.state == STATE_TEST:
                self.state = self.__test_solver()
            elif self.state == STATE_DONE:
                ros.signal_shutdown("Done!")
                break
            self.rate.sleep()

    def __startup(self):
        self.sophie.move_to_joint_configuration(SOPHIE_REST)
        self.wilson.move_to_joint_configuration(WILSON_REST)
        return STATE_TEST

    def __measure(self):
        time.sleep(0.5)
        x0 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        pose = np.array([+0.00, -1.00, +0.50, -0.50, -0.00, +0.00]) * np.pi
        self.sophie.move_to_joint_configuration(pose)
        time.sleep(0.5)
        x1 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        wrist3_tcp = abs(x1[0] - x0[0])
        pyout(f"Wrist3 -> TCP = {wrist3_tcp:.5f}")

        pose = np.array([+0.00, -1.00, +0.50, -0.00, -0.00, +0.00]) * np.pi
        self.sophie.move_to_joint_configuration(pose)
        time.sleep(0.5)
        x2 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        wrist2_wrist3 = abs(x2[2] - x1[2])
        pyout(f"Wrist2 -> Wrist1 = {wrist2_wrist3:.5f}")

        pose = np.array([+0.00, -0.50, +0.00, -0.50, -0.00, +0.00]) * np.pi
        self.sophie.move_to_joint_configuration(pose)
        time.sleep(0.5)
        x3 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        shoulder_elbow = abs(x3[2] - x1[2])
        pyout(f"Shoulder -> Elbow = {shoulder_elbow:.5f}")

        pose = np.array([+0.00, -0.50, +0.50, -1.00, -0.00, +0.00]) * np.pi
        self.sophie.move_to_joint_configuration(pose)
        time.sleep(0.5)
        x4 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        elbow_wrist1 = abs(x4[0] - x3[0])
        pyout(f"Elbow -> wrist1 = {elbow_wrist1:.5f}")

        origin_base = x3[2] - wrist2_wrist3 - elbow_wrist1 - shoulder_elbow
        pyout(f"Origin -> Base = {origin_base: .5f}")

        pose = np.array([+0.50, -0.50, +0.00, -0.50, -0.50, +0.00]) * np.pi
        self.sophie.move_to_joint_configuration(pose)
        time.sleep(0.5)
        x5 = self.sophie.get_tcp_pose()[:3, 3]
        # pyout(self.sophie.get_tcp_pose())

        wrist2_offset = abs(x3[1] - x5[1])
        pyout(f"Wrist2 offset = {wrist2_offset:.5f}")

        # shoulder_elbow = abs(x3[2] - x1[2])
        # pyout(f"Shoulder -> Elbow = {shoulder_elbow:.3f}")

        return STATE_DONE

    def __test_solver(self):
        from pairo_butler.ur3_arms.ur3_solver import (
            PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3,
        )
        from pairo_butler.ur3_arms.ur3_solver import LENGHT_ORIGIN_TO_BASE
        from pairo_butler.ur3_arms.ur3_solver import LENGTH_BASE_TO_ELBOW
        from pairo_butler.ur3_arms.ur3_solver import LENGTH_ELBOW_TO_WRIST1
        from pairo_butler.ur3_arms.ur3_solver import LENGTH_WRIST2_TO_WRIST3
        from pairo_butler.ur3_arms.ur3_solver import LENGTH_WRIST3_TO_TOOL

        # self.sophie.grasp_down(np.array([0.5, 0.13301, 0.5]))

        # X = np.array(
        #     [
        #         -LENGTH_ELBOW_TO_WRIST1 - LENGTH_WRIST2_TO_WRIST3,
        #         -PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3,
        #         LENGHT_ORIGIN_TO_BASE + LENGTH_BASE_TO_ELBOW - LENGTH_WRIST3_TO_TOOL,
        #     ]
        # )
        # self.sophie.grasp_down(X)

        X = np.array([-0.3, 0.0, 0.69])
        for z in (
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ):
            self.sophie.grasp_horizontal(X, z)

        X = np.array(
            [
                -LENGTH_ELBOW_TO_WRIST1 - LENGTH_WRIST3_TO_TOOL,
                -PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3,
                LENGHT_ORIGIN_TO_BASE + LENGTH_BASE_TO_ELBOW + LENGTH_WRIST2_TO_WRIST3,
            ]
        )
        z = np.array([-1.0, 0.0, 0.0])

        self.sophie.grasp_horizontal(X, z)

        X = np.linspace(-0.25, +0.25, num=2, endpoint=True)
        Y = np.linspace(-0.5, 0.5, num=3, endpoint=True)
        for x in X:
            for y in Y:
                try:
                    self.sophie.grasp_down(np.array([x, y, 0.2]))
                    self.sophie.grasp_down(np.array([x, y, 0.01]))
                    self.sophie.close_gripper()
                    time.sleep(1)
                    self.sophie.open_gripper()
                except AssertionError:
                    pyout("Cannot reach (x,y)")
            self.sophie.move_to_joint_configuration(SOPHIE_REST)

        return STATE_DONE


def main():
    node = UR3Measurer()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
