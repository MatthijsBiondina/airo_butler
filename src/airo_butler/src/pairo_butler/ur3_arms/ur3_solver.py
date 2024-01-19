import time
import numpy as np
from pairo_butler.utils.shapes import (
    compute_triangle_angles_sss,
    find_intersections_between_circles,
)

from pairo_butler.utils.tools import degree_string, pyout


PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3 = 0.13301
FORBIDDEN_COLUMN_RADIUS = 0.01 + PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3

LENGHT_ORIGIN_TO_BASE = 0.16521
LENGTH_BASE_TO_ELBOW = 0.42497
LENGTH_ELBOW_TO_WRIST1 = 0.38978
LENGTH_WRIST2_TO_WRIST3 = 0.09941
LENGTH_WRIST3_TO_TOOL = 0.27287


# L_BASE_OFFST = 0.000
# L_FLOOR2BASE = 0.000
# L_BASE2ELBOW = 0.5
# L_ELBOW2WRST = 0.5
# L_WRIST2HAND = 0.000
# L_HAND2GRIPR = 0.000


class UR3WilsonSolver:
    def __init__(self) -> None:
        pass


class UR3SophieSolver:
    def __init__(self) -> None:
        pass

    def solve_tcp_horizontal(self, tool_xyz, z_axis, flipped=False, allow_flip=True):
        assert z_axis[-1] == 0, "Camera z-axis must be horizontal"
        # Normalize z-axis
        z_axis /= np.linalg.norm(z_axis)

        # Initialize joint configuration
        joint_config = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, +0.00]) * np.pi

        # Calculate wrist3 point
        wrist3 = tool_xyz - z_axis * LENGTH_WRIST3_TO_TOOL

        # Calculate wrist2 point
        if flipped:
            wrist2 = wrist3 + np.array([0.0, 0.0, 1.0]) * LENGTH_WRIST2_TO_WRIST3
        else:
            wrist2 = wrist3 - np.array([0.0, 0.0, 1.0]) * LENGTH_WRIST2_TO_WRIST3

        # Calculate wrist1 point. The orthogonal projection of wrist2 onto the radius
        # of the circle around which the base spins.
        distance_wrist2_projection_on_xy_plane = np.linalg.norm(wrist2[:2])
        wrist2_offset_angle = np.arcsin(
            PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3
            / distance_wrist2_projection_on_xy_plane
        )

        if np.isnan(wrist2_offset_angle):
            assert (
                distance_wrist2_projection_on_xy_plane < FORBIDDEN_COLUMN_RADIUS
            ), "Fail case for changing approach angle triggered for the wrong reason"
            # Wrist 2 cannot reach above the base, because of morphology.
            # So we move wrist2 over to the edge of the forbidden column (w2')
            #
            #                   tool
            #                    /\
            #                   /||\
            #                    ||
            #                 /  ||  \
            #                /   ||   \
            #                    ||
            #         @    /   ######   \    @
            #          @@ /####  ||  ####\ @@
            #    w2' ->  @#      ||      #@  <- w2'
            #          ##  @@@@  ||  @@@@  ##
            #         #        @ w2 @         #
            #        #                        #
            #        #           BA           #
            #        #           SE           #
            #        #                        #
            #         #                      #
            #          ##                  ##
            #            ##              ##
            #              ####      ####
            #                  ######

            # (1) Everything happening below is a projection in the xy plane (1)
            # Find intersections between the forbidden circle and the circle around
            # tool with radius LENGTH_WRIST3_TO_TOOL
            intersections = find_intersections_between_circles(
                x0=0.0,
                y0=0.0,
                r0=FORBIDDEN_COLUMN_RADIUS,
                x1=tool_xyz[0],
                y1=tool_xyz[1],
                r1=LENGTH_WRIST3_TO_TOOL,
            )
            assert len(intersections) == 2
            int0, int1 = tuple(intersections)

            tool_to_inter0 = np.array(int0) - tool_xyz[:2]
            tool_to_inter1 = np.array(int1) - tool_xyz[:2]
            tool_to_wrist2 = wrist2[:2] - tool_xyz[:2]

            angle0 = np.arctan2(
                tool_to_wrist2[1] - tool_to_inter0[1],
                tool_to_wrist2[0] - tool_to_inter0[0],
            )
            angle1 = np.arctan2(
                tool_to_wrist2[1] - tool_to_inter1[1],
                tool_to_wrist2[0] - tool_to_inter1[0],
            )

            z_axis_new = z_axis
            if abs(angle0) < abs(angle1):
                z_axis_new[:2] = -tool_to_inter0 / np.linalg.norm(tool_to_inter0)
            else:
                z_axis_new[:2] = -tool_to_inter1 / np.linalg.norm(tool_to_inter1)

            return self.solve_tcp_horizontal(tool_xyz=tool_xyz, z_axis=z_axis_new)

        # rotate wrist2 around base
        wrist2_rotated_projection_xy = np.array(
            [
                wrist2[0] * np.cos(-wrist2_offset_angle)
                - wrist2[1] * np.sin(-wrist2_offset_angle),
                wrist2[0] * np.sin(-wrist2_offset_angle)
                + wrist2[1] * np.cos(-wrist2_offset_angle),
            ]
        )
        # ... and determine radius
        distance_base_to_wrist1 = (
            np.cos(wrist2_offset_angle) * distance_wrist2_projection_on_xy_plane
        )
        # Then project onto the circle with computed distance from base.
        wrist1 = np.array([np.nan, np.nan, wrist2[2]])
        wrist1[:2] = (
            wrist2_rotated_projection_xy
            / np.linalg.norm(wrist2_rotated_projection_xy)
            * distance_base_to_wrist1
        )

        # Compute base joint angle
        base_angle = np.arctan2(wrist1[1], wrist1[0]) % (2 * np.pi) - np.pi
        joint_config[0] = base_angle

        # Calculate the elbow angle, based on known distances between base, elbow, and
        # wrist1.
        base = np.array([0.0, 0.0, LENGHT_ORIGIN_TO_BASE])
        distance_base_to_wrist1 = np.linalg.norm(wrist1 - base)
        elbow_angle, _, angle_elbow_base_wrist1 = compute_triangle_angles_sss(
            distance_base_to_wrist1, LENGTH_BASE_TO_ELBOW, LENGTH_ELBOW_TO_WRIST1
        )
        joint_config[2] = np.pi - elbow_angle

        # Determine intermediate point; (-1, 0, height_base) rotated around z by
        # base angle
        intermediate_point = np.array(
            [-np.cos(base_angle), -np.sin(base_angle), LENGHT_ORIGIN_TO_BASE]
        )

        # Compute angle of wrist2.
        # Based on projections of wrist1, wrist3, and tcp in xy-plane compute angle
        # wrist1 <- wrist3 -> tool.
        wrist3_angle_unsigned, _, _ = compute_triangle_angles_sss(
            np.linalg.norm(tool_xyz[:2] - wrist1[:2]),
            LENGTH_WRIST3_TO_TOOL,
            PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3,
        )
        # To determine sign - forward or backwards - compute
        # transposed tcp' if the tool was attached to wrist1 instead of wrist3, and
        # compute angle intermediate_point <- wrist1 -> tcp'. If smaller than 90
        # degrees it points forward, otherwise backwards.
        tool_ = tool_xyz + (wrist1 - wrist3)
        angle_intermediate_wrist1_tool_unsigned, _, _ = compute_triangle_angles_sss(
            np.linalg.norm(tool_[:2] - intermediate_point[:2]),
            LENGTH_WRIST3_TO_TOOL,
            np.linalg.norm(wrist1[:2] - intermediate_point[:2]),
        )
        if np.rad2deg(wrist3_angle_unsigned) < 90:
            wrist3_angle = -angle_intermediate_wrist1_tool_unsigned
        else:
            wrist3_angle = angle_intermediate_wrist1_tool_unsigned
        wrist3_angle = (wrist3_angle + np.pi) % (2 * np.pi) - np.pi

        if flipped:
            joint_config[4] = 0.5 * np.pi - wrist3_angle
        else:
            joint_config[4] = wrist3_angle - 0.5 * np.pi

        # Compute the angle between base -> wrist1 and the xy plane
        # This angle can be larger than 90 degrees if we lean backwards
        angle_base_wrist1_with_xy_plane, _, _ = compute_triangle_angles_sss(
            np.linalg.norm(wrist1 - intermediate_point),
            distance_base_to_wrist1,
            np.linalg.norm(intermediate_point[:2]),
        )

        # Compute shoulder angle, which is the sum of ...
        # > the angle betwen the xy-plane and base -> wrist1
        # > the angle between base -> wrist1 and base -> elbow
        shoulder_angle = angle_base_wrist1_with_xy_plane + angle_elbow_base_wrist1
        joint_config[1] = -shoulder_angle

        # Compute the angle of wrist2, which compensates for the shoulder and elbow angle,
        # So that the gripper is horizontal
        # todo: points down now probably +0.5 or something like that
        wrist2_angle = joint_config[1] + joint_config[2]
        if flipped:
            joint_config[3] = -wrist2_angle
        else:
            joint_config[3] = -np.pi - wrist2_angle

        # Compute tcp
        tcp = np.array(
            [
                [np.nan, 0.0, np.nan, tool_xyz[0]],
                [np.nan, 0.0, np.nan, tool_xyz[1]],
                [0.0, -1.0, 0.0, tool_xyz[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        tcp[:2, 2] = z_axis[:2]  # third element must be 0 (see assertion earlier)
        tcp[:3, 0] = np.cross(tcp[:3, 1], tcp[:3, 2])

        # Check whether the hand should be flipped (wrist3 below wrist2)
        if not flipped and np.rad2deg(wrist2_angle) > 15 and allow_flip:
            return self.solve_tcp_horizontal(
                tool_xyz=tool_xyz, z_axis=z_axis, flipped=True
            )

        # Do some checks to avoid collision
        assert (
            np.rad2deg(elbow_angle) > 22
        ), f"Bending elbow {np.rad2deg(elbow_angle):.1f} degrees risks self-collision."
        assert wrist2[2] > 0.08, f"Wrist too close to table ({wrist2[2]*100:.1f} cm)"

        return tcp, joint_config, flipped

    def solve_tcp_vertical_down(self, X_target):
        """
            a: Base
            b: Shoulder
            c: Elbow
            d:   "
            e: Wrist 1
            f: Wrist 2
            g: Wrist 3
            (h): Point aligned with a and g from bird-eye-view


            #####
        --- # g #  (h)
            #####   .
             ###    .
            ##### #####
            # f ### e #
            ##### #####
                   ###
                   ###
                   ###
                   ###
                   ###
                   ###
            ##### #####
            # c ### d #
            ##### #####                         #####
             ###                                # d #
             ###                               #########
             ###                              ##       ####
             ###                             ##           ####
             ###                            ##               #######
             ###                           ##            . . . # e #
            ##### #####                #####       . . .       #####
            # b ### a #                # a # . . .
            ##### #####                #####
                   ###
                 #######
        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Initialize joint configuration
        joint_config = np.array([np.nan, np.nan, np.nan, np.nan, -0.50, +0.00]) * np.pi

        # Calculate target point above the gripper
        wrist3 = X_target + np.array([0.0, 0.0, LENGTH_WRIST3_TO_TOOL])

        # Compute intermediate point; the orthogonal projection of wrist3 onto the radius of
        # the circle around which the base spins.
        distance_wrist3_projection_on_xy_plane = np.linalg.norm(wrist3[:2])
        wrist3_offset_angle = np.arcsin(
            PERPENDICULAR_TRANSLATION_BASE_TO_WRIST3
            / distance_wrist3_projection_on_xy_plane
        )
        # First rotate wrist3 around the base
        wrist3_rotated_projection_xy = np.array(
            [
                wrist3[0] * np.cos(-wrist3_offset_angle)
                - wrist3[1] * np.sin(-wrist3_offset_angle),
                wrist3[0] * np.sin(-wrist3_offset_angle)
                + wrist3[1] * np.cos(-wrist3_offset_angle),
            ]
        )
        # And determine radius
        distance_base_intermediate_point1 = (
            np.cos(wrist3_offset_angle) * distance_wrist3_projection_on_xy_plane
        )
        # Then project onto the circle with computed distance from base.
        intermediate_point1 = np.array([np.nan, np.nan, wrist3[2]])
        intermediate_point1[:2] = (
            wrist3_rotated_projection_xy
            / np.linalg.norm(wrist3_rotated_projection_xy)
            * distance_base_intermediate_point1
        )

        # Compute base joint angle
        base_angle = (
            np.arctan2(intermediate_point1[1], intermediate_point1[0]) % (2 * np.pi)
            - np.pi
        )
        joint_config[0] = base_angle

        # Calculate position of wrist1.
        # wrist2 -> wrist3 is parallel to the xy-plane. Thus, we find wrist1 by subtracting the
        # lenght of wrist2 -> wrist 3 from base -> intermediate point.
        # Calculate what portion of base -> intermediate point is covered by wrist2 -> wrist3
        ratio = LENGTH_WRIST2_TO_WRIST3 / (
            np.cos(wrist3_offset_angle) * distance_wrist3_projection_on_xy_plane
        )
        wrist1 = np.copy(intermediate_point1)
        wrist1[:2] *= 1 - ratio

        # Calculate elbow angle.
        # We can calculate the distance between base and wrist1, which closes the triangle [base,
        # elbow, wrist1]. Since we know the lengths of all edges in this triangle, we can compute,
        # the angle of the elbow joint.
        base = np.array([0.0, 0.0, LENGHT_ORIGIN_TO_BASE])
        distance_base_to_wrist1 = np.linalg.norm(wrist1 - base)
        elbow_angle, _, angle_elbow_base_wrist1 = compute_triangle_angles_sss(
            distance_base_to_wrist1, LENGTH_BASE_TO_ELBOW, LENGTH_ELBOW_TO_WRIST1
        )
        joint_config[2] = np.pi - elbow_angle

        # Determine intermediate point 2. This point is (-1, 0, height_base) rotated around z by
        # base_angle.
        intermediate_point2 = np.array(
            [-np.cos(base_angle), -np.sin(base_angle), LENGHT_ORIGIN_TO_BASE]
        )

        # Compute the angle between base -> wrist1 and the xy plane
        # This angle can be larger than 90 degrees if we lean very far backwards. That's why we
        # use intermediate_point2
        angle_base_wrist1_with_xy_plane, _, _ = compute_triangle_angles_sss(
            np.linalg.norm(wrist1 - intermediate_point2),
            distance_base_to_wrist1,
            np.linalg.norm(intermediate_point2[:2]),
        )

        # Compute shoulder angle, which is the sum of ...
        # > the angle between the xy-plane and base->wrist1
        # > the angle betwen the base->wrist1 and base->elbow segments
        shoulder_angle = angle_base_wrist1_with_xy_plane + angle_elbow_base_wrist1

        joint_config[1] = -shoulder_angle

        # Compute the angle of wrist2, which compensates for the shoulder and elbow angle,
        # so that the gripper points down
        wrist2_angle = joint_config[1] + joint_config[2]
        joint_config[3] = -0.5 * np.pi - wrist2_angle

        # Compute tcp
        tcp = np.array(
            [
                [np.nan, np.nan, 0.0, X_target[0]],
                [np.nan, np.nan, 0.0, X_target[1]],
                [0.0, 0.0, -1.0, X_target[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        tcp[:2, 1] = -intermediate_point1[:2] / np.linalg.norm(intermediate_point1[:2])
        tcp[:3, 0] = np.cross(tcp[:3, 1], tcp[:3, 2])

        # And do some checks to avoid collision
        assert (
            np.rad2deg(elbow_angle) > 22
        ), f"Bending elbow {np.rad2deg(elbow_angle):.1f} degrees risks self-collision."

        return tcp, joint_config
