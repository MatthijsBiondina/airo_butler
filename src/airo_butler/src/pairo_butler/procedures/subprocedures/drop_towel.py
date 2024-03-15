import numpy as np
from pairo_butler.procedures.subprocedure import Subprocedure


np.set_printoptions(precision=2, suppress=True)


class DropTowel(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        self.__sophie_let_go_if_both_holding_towel()
        self.__move_to_rest_pose()
        self.__let_go_sophie()
        self.__let_go_wilson()
        self.__move_to_rest_pose()

    def __move_to_rest_pose(self):
        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_rest_sophie),
            wilson=np.deg2rad(self.config.joints_rest_wilson),
        )
        self.sophie.execute_plan(plan)

    def __sophie_let_go_if_both_holding_towel(self):
        if (
            self.sophie.get_gripper_width() < 0.03
            and self.wilson.get_gripper_width() < 0.03
        ):
            self.sophie.open_gripper()

    def __let_go_sophie(self):
        if self.sophie.get_gripper_width() < 0.03:
            try:
                plan = self.ompl.plan_to_joint_configuration(
                    sophie=np.deg2rad(self.config.joints_hold_sophie)
                )
                self.sophie.execute_plan(plan)

                plan = self.ompl.plan_to_tcp_pose(sophie=self.config.tcp_drop)
                self.sophie.execute_plan(plan)
            except RuntimeError:
                pass
            self.sophie.open_gripper()
            self.__move_to_rest_pose()

    def __let_go_wilson(self):
        if self.wilson.get_gripper_width() < 0.03:
            try:
                plan = self.ompl.plan_to_tcp_pose(wilson=self.config.tcp_drop)
                self.wilson.execute_plan(plan)
            except RuntimeError:
                pass
            self.wilson.open_gripper()
            self.__move_to_rest_pose()
