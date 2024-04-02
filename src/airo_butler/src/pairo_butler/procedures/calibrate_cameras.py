from pairo_butler.procedures.subprocedures.calibrate_wilson_zed import (
    CalibrateWilsonZed,
)
from pairo_butler.procedures.subprocedures.wilson_recieve_charuco import (
    WilsonRecieveCharuco,
)
from pairo_butler.procedures.subprocedures.startup import Startup
from pairo_butler.procedures.machine import Machine
from pairo_butler.motion_planning.ompl_client import OMPLClient
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.utils.tools import load_config, pyout


class CalibrateMachine(Machine):
    def __init__(self, name: str = "unfold_machine"):
        super().__init__(name)

    def run(self):
        Startup(drop=False, **self.kwargs).run()
        WilsonRecieveCharuco(**self.kwargs).run()
        CalibrateWilsonZed(**self.kwargs).run()
        pyout()


def main():
    node = CalibrateMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
