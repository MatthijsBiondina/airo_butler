import numpy as np
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
import rospy as ros


class Fling(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        plan = self.ompl.plan_to_joint_configuration(
            sophie=np.deg2rad(self.config.joints_scan1_sophie), scene="default"
        )
        self.sophie.execute_plan(plan)

        plan = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_fling1_wilson)
        )
        self.wilson.execute_plan(plan)

        plan1 = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_fling2_wilson)
        )
        self.wilson.execute_plan(plan1)

        plan2 = self.ompl.plan_to_joint_configuration(
            wilson=np.deg2rad(self.config.joints_fling1_wilson)
        )
        self.wilson.execute_plan(plan2)

        ros.sleep(0.1)
        self.wilson.execute_plan(plan1)
        ros.sleep(0.1)
        self.wilson.execute_plan(plan2)

        pyout()
