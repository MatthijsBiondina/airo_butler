from typing import Optional
import numpy as np
from pairo_butler.ur3_arms.planner.dual_ur5e import get_robot_diagram_builder_dual_ur5e
from pairo_butler.ur3_arms.planner.collisions import get_collision_checker
from pairo_butler.utils.tools import pyout
from airo_robots.manipulators import URrtde


class Planner:
    def __init__(self, left_arm: URrtde, right_arm: URrtde) -> None:
        # assumptions, left is right, right is left
        self.sophie = right_arm
        self.wilson = left_arm

        current_joints_sophie = self.sophie.get_joint_configuration()
        current_joints_wilson = self.wilson.get_joint_configuration()
        current_joints = np.concatenate((current_joints_sophie, current_joints_wilson))
        (
            self.robot_diagram_builder,
            self.robot_indexes,
        ) = get_robot_diagram_builder_dual_ur5e()
        self.plant = self.robot_diagram_builder.plant()
        self.diagram = self.robot_diagram_builder.Build()
        self.collision_checker = get_collision_checker(self.diagram, self.robot_indexes)
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.diagram.ForcedPublish(self.context)
        self.plant.SetPositions(self.plant_context, current_joints)
        self.diagram.ForcedPublish(self.context)

    def plan_to_joint_configuration(
        self, wilson: Optional[np.ndarray] = None, sophie: Optional[np.ndarray] = None
    ):
        sophie = sophie if sophie is not None else self.sophie.get_joint_configuration()
        wilson = wilson if wilson is not None else self.wilson.get_joint_configuration()

        pyout()
