from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure


class CalibrateWilsonZed(Subprocedure):
    SCENE = "wilson_holds_charuco"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        plan = self.ompl.plan_to_tcp_pose(
            wilson=self.config.tcp_calibration_wilson, scene=self.SCENE
        )
        self.wilson.execute_plan(plan)

        pyout()
