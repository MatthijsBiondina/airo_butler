from pairo_butler.procedures.subprocedures.drop_charuco_board import DropCharucoBoard
from pairo_butler.procedures.subprocedure import Subprocedure
import rospy as ros


class WilsonRecieveCharuco(Subprocedure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.sophie.get_gripper_width() < 0.002:
            self.sophie.open_gripper()

        self.wilson.close_gripper()
        if self.wilson.get_gripper_width() < 0.002:
            ros.loginfo("Give the Charuco board to Wilson.")
        while not ros.is_shutdown() and self.wilson.get_gripper_width() < 0.002:
            self.wilson.open_gripper()
            self.wilson.close_gripper()
            ros.sleep(1.0)
