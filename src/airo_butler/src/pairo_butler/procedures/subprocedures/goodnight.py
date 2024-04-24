import numpy as np
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure


class Goodnight(Subprocedure):
    def __init__(self, drop: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.drop = drop
        self.kwargs = kwargs

    def run(self):
        DropTowel(**self.kwargs).run()

        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_sleep_sophie),
            wilson=np.deg2rad(self.config.joints_sleep_wilson),
        )
        self.sophie.execute_plan(plan)
