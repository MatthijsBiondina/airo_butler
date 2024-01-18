from functools import partial
from typing import Optional
import numpy as np
from pairo_butler.ur3_arms.planner.ompl_state_space import (
    function_numpy_to_ompl,
    numpy_to_ompl_state,
    single_arm_state_space,
)
from pairo_butler.ur3_arms.planner.dual_ur5e import get_robot_diagram_builder_dual_ur5e
from pairo_butler.ur3_arms.planner.collisions import get_collision_checker
from pairo_butler.utils.tools import pyout
from airo_robots.manipulators import URrtde

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

ou.setLogLevel(ou.LOG_DEBUG)


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
        self,
        wilson_pose: Optional[np.ndarray] = None,
        sophie_pose: Optional[np.ndarray] = None,
    ):
        sophie_pose = (
            sophie_pose
            if sophie_pose is not None
            else self.sophie.get_joint_configuration()
        )
        wilson_pose = (
            wilson_pose
            if wilson_pose is not None
            else self.wilson.get_joint_configuration()
        )

        space = single_arm_state_space()

        start_state_sophie = numpy_to_ompl_state(
            self.sophie.get_joint_configuration(), space
        )
        goal_state_sophie = numpy_to_ompl_state(sophie_pose, space)

        print(start_state_sophie)
        print(goal_state_sophie)

        def are_joints_collision_free(joints_left, joints_right) -> bool:
            joints = np.concatenate((joints_left, joints_right))
            return self.collision_checker.CheckConfigCollisionFree(joints)

        pyout(
            are_joints_collision_free(
                self.sophie.get_joint_configuration(),
                self.wilson.get_joint_configuration(),
            )
        )

        are_joints_sophie_collision_free = partial(
            are_joints_collision_free,
            joints_right=self.wilson.get_joint_configuration(),
        )
        pyout(are_joints_sophie_collision_free(self.sophie.get_joint_configuration()))

        is_state_valid = function_numpy_to_ompl(are_joints_sophie_collision_free, 6)
        pyout(is_state_valid(start_state_sophie))
        pyout(is_state_valid(goal_state_sophie))

        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        simple_setup.setStartAndGoalStates(start_state_sophie, goal_state_sophie)

        step = float(np.deg2rad(5))
        resolution = step / space.getMaximumExtent()
        simple_setup.getSpaceInformation().setStateValidityCheckingResolution(
            resolution
        )

        planner = og.RRTstar(simple_setup.getSpaceInformation())
        simple_setup.setPlanner(planner)

        pyout("Start Solving")

        try:
            simple_setup.solve(1.0)
        except Exception as e:
            pyout("An exception")

        pyout("Solve Finished")

        pyout()
