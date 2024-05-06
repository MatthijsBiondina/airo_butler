import numpy as np
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure


class Startup(Subprocedure):
    def __init__(self, drop: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.drop = drop
        self.kwargs = kwargs

    def run(self):
        if self.drop:
            DropTowel(**self.kwargs).run()
        else:
            scene = "default"
            if self.wilson.get_gripper_width() < 0.03:
                scene = "wilson_holds_charuco"
            elif self.sophie.get_gripper_width() < 0.03:
                scene = "sophie_holds_charuco"

            plan = self.ompl.plan_to_joint_configuration(
                sophie=np.deg2rad(self.config.joints_rest_sophie),
                wilson=np.deg2rad(self.config.joints_rest_wilson),
                scene=scene,
            )
            
            self.sophie.execute_plan(plan)
