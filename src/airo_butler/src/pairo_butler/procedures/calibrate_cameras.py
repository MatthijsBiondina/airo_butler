from pairo_butler.procedures.subprocedures.drop_charuco_board import DropCharucoBoard
from pairo_butler.procedures.subprocedures.calibrate_sophie_zed import (
    CalibrateSophieZed,
)
from pairo_butler.procedures.subprocedures.transfer_charuco_board import (
    TransferCharucoBoard,
)
from pairo_butler.procedures.subprocedures.calibrate_sophie_rs2 import (
    CalibrateSophieRS2,
)
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
        CalibrateSophieRS2(**self.kwargs).run()
        TransferCharucoBoard(**self.kwargs).run()
        CalibrateSophieZed(**self.kwargs).run()
        DropCharucoBoard(**self.kwargs).run()


def main():
    node = CalibrateMachine()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
