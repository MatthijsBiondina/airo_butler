import pickle
from typing import List
import numpy as np
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.utils.transformations_3d import homogenous_transformation
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import pyout
from pairo_butler.procedures.subprocedure import Subprocedure
from airo_butler.msg import PODMessage
import rospy as ros


class GraspCorner(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.keypoint_tcps = self.__choose_corner(KalmanFilter.get_state())

    def run(self):
        try:
            self.sophie.set_gripper_width(0.05)
            for keypoint in self.keypoint_tcps:
                for grasp_tcp, pregrasp_tcp in self.__compute_grasps_and_pregrasps(
                    keypoint
                ):
                    try:
                        tcp_path = self.__compute_intermediate_tcps(
                            pregrasp_tcp, grasp_tcp
                        )
                        if self.__perform_shortest_grasp(tcp_path):
                            self.sophie.close_gripper()
                            raise BreakException
                        else:
                            continue
                    except RuntimeError as e:
                        ros.logwarn(f"grasp_corner.py | {e}")
                        continue
        except BreakException:
            return True

        return False

    def __choose_corner(self, state, threshold=1e-11):
        covariances = state.covariances
        uncertainty = np.linalg.det(covariances)

        indexes_by_certainty = np.argsort(uncertainty)

        uncertainty = uncertainty[indexes_by_certainty]
        for c in uncertainty:
            pyout(f"{c:e}")
        means = state.means[indexes_by_certainty]

        means = means[uncertainty < threshold]

        return self.__convert_kalman_states_to_tcps(means)

    def __compute_grasps_and_pregrasps(self, kp):
        # Determine whether the x-axis of the keypoint tcp points inwards or outwards
        # relative to the rest of the towel
        kp_relative_to_wilson = kp[:3, 3] - self.wilson.get_tcp_pose()[:3, 3]

        x_outwards = np.dot(kp[:3, 0], kp_relative_to_wilson) > 0

        if x_outwards:  # x points outwards
            tcp_scoop = np.eye(4)
            tcp_scoop[:3, 0] = (kp[:3, 0] - kp[:3, 1]) / np.linalg.norm(
                kp[:3, 0] - kp[:3, 1]
            )
            tcp_scoop[:3, 1] = kp[:3, 2]
            tcp_scoop[:3, 2] = (-kp[:3, 0] - kp[:3, 1]) / np.linalg.norm(
                -kp[:3, 0] - kp[:3, 1]
            )
            tcp_scoop[:3, 3] = kp[:3, 3] - 0.03 * kp[:3, 2]
        else:
            tcp_scoop = np.eye(4)
            tcp_scoop[:3, 0] = (-kp[:3, 0] - kp[:3, 1]) / np.linalg.norm(
                kp[:3, 0] - kp[:3, 1]
            )
            tcp_scoop[:3, 1] = -kp[:3, 2]
            tcp_scoop[:3, 2] = (kp[:3, 0] - kp[:3, 1]) / np.linalg.norm(
                -kp[:3, 0] - kp[:3, 1]
            )
            tcp_scoop[:3, 3] = kp[:3, 3] - 0.03 * kp[:3, 2]
            ros.logwarn(f"[grasp_corner.py] Approach from this side is untested")

        grasp_scoop = self.__compute_grasp(tcp_scoop)
        pregrasp_scoop = self.__compute_pregrasp(tcp_scoop)

        if not (grasp_scoop is None or pregrasp_scoop is None):
            yield (grasp_scoop, pregrasp_scoop)

        # Flip 180 degrees to account for self-collision with the realsense
        tcp_scoop_flipped = tcp_scoop @ homogenous_transformation(yaw=180)
        grasp_scoop_flipped = self.__compute_grasp(tcp_scoop_flipped)
        pregrasp_scoop_flipped = self.__compute_pregrasp(tcp_scoop_flipped)
        if not (grasp_scoop_flipped is None or pregrasp_scoop_flipped is None):
            yield (grasp_scoop_flipped, pregrasp_scoop_flipped)

        # If those don't work, try some random perturbations
        for ii in range(10):
            tcp_modified = tcp_scoop.copy()
            tcp_modified = tcp_modified @ homogenous_transformation(
                roll=np.random.randint(-20, 20),
                pitch=np.random.randint(-5, 5),
                yaw=np.random.randint(-20, 20),
            )
            grasp_modified = self.__compute_grasp(tcp_modified)
            pregrasp_modified = self.__compute_pregrasp(tcp_modified)
            if not (grasp_modified is None or pregrasp_modified is None):
                yield (grasp_modified, pregrasp_modified)

        # Make poses when approaching from front (kp z-axis)
        tcp_front = kp @ homogenous_transformation(pitch=180)

        for theta in np.linspace(0, 45, num=10)[::-1]:
            if x_outwards:
                tcp_mod = tcp_front @ homogenous_transformation(pitch=theta)
            else:
                tcp_mod = tcp_front @ homogenous_transformation(pitch=-theta)

            grasp_front = self.__compute_grasp(tcp_mod)
            pregrasp_front = self.__compute_pregrasp(tcp_mod)
            if not (grasp_front is None or pregrasp_front is None):
                yield (grasp_front, pregrasp_front)

    def __compute_grasp(
        self, kp_tcp: np.ndarray, overshoot_distance=0.03
    ) -> np.ndarray:
        grasp = kp_tcp.copy()
        grasp[:3, 3] += grasp[:3, 2] * overshoot_distance
        if self.__validate_pose(grasp):
            return grasp
        else:
            return None

    def __compute_pregrasp(self, grasp: np.ndarray, min_distance=0.1) -> np.ndarray:
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= pregrasp[:3, 2] * min_distance
        for _ in range(10):
            if self.__validate_pose(pregrasp, scene="hanging_towel"):
                break
            pregrasp[:3, 3] -= pregrasp[:3, 2] * 0.05
        else:
            return None

        return pregrasp

    def __validate_pose(self, tcp: np.ndarray, scene: str = "default") -> bool:
        ik_solutions = self.ompl.get_ik_solutions(sophie=tcp, scene=scene)
        return ik_solutions.size > 0

    def __convert_kalman_states_to_tcps(self, means: np.ndarray):
        keypoints = []
        for kp in means:
            keypoints.append(
                np.array(
                    [
                        [np.sin(kp[3]), 0.0, np.cos(kp[3]), kp[0]],
                        [-np.cos(kp[3]), 0.0, np.sin(kp[3]), kp[1]],
                        [0.0, -1.0, 0.0, kp[2]],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )

        return keypoints

    def __compute_intermediate_tcps(
        self, pregrasp: np.ndarray, grasp: np.ndarray, max_step_size: float = 0.01
    ):
        num = 10
        while not ros.is_shutdown():
            alpha = np.linspace(start=0, stop=1, num=num, endpoint=True)[:, None, None]
            path = (1 - alpha) * pregrasp[None, ...] + alpha * grasp[None, ...]
            if np.linalg.norm(path[0, :3, 3] - path[1, :3, 3]) < max_step_size:
                break
            num += 1

        return path

    def __perform_shortest_grasp(self, tcps: np.ndarray) -> np.ndarray:

        joints = []
        for tcp in tcps:
            ik_solutions = self.ompl.get_ik_solutions(sophie=tcp)

            if ik_solutions.size == 0:
                return None
            joints.append(ik_solutions)

        shortest_path = self.__compute_shortest_path(joints)

        if shortest_path is None:
            return None

        plan = self.ompl.plan_to_joint_configuration(
            sophie=shortest_path[0], scene="hanging_towel"
        )
        self.sophie.execute_plan(plan)

        plan = self.ompl.toppra(sophie=shortest_path)
        if plan is None:
            return False
        else:
            self.sophie.execute_plan(plan)
            return True

    def __compute_shortest_path(self, joints: List[np.ndarray]):
        nodes = []
        for step in range(len(joints) - 1, -1, -1):
            for joint_pose in joints[step]:
                nodes.append(
                    {
                        "pose": joint_pose,
                        "step": step,
                        "children": [n for n in nodes if n["step"] == step + 1],
                        "estimated_distance": np.min(
                            np.linalg.norm(joint_pose[None, :] - joints[-1], axis=1)
                        ),
                    }
                )

        optimal_path = None
        unfinished_paths = [
            {"nodes": [n], "length": 0, "min_cost": n["estimated_distance"]}
            for n in nodes
            if n["step"] == 0
        ]
        unfinished_paths = sorted(unfinished_paths, key=lambda path: path["min_cost"])

        while not ros.is_shutdown():
            old_path = unfinished_paths.pop(0)

            if (
                optimal_path is not None
                and old_path["min_cost"] >= optimal_path["length"]
            ):
                break

            if len(old_path["nodes"][-1]["children"]):
                for child_node in old_path["nodes"][-1]["children"]:
                    new_path = {
                        "nodes": old_path["nodes"] + [child_node],
                        "length": old_path["length"]
                        + np.linalg.norm(
                            old_path["nodes"][-1]["pose"] - child_node["pose"],
                        ),
                    }
                    new_path["min_cost"] = (
                        new_path["length"] + child_node["estimated_distance"]
                    )

                    insertion_idx = 0
                    try:
                        while (
                            new_path["min_cost"]
                            > unfinished_paths[insertion_idx]["min_cost"]
                        ):
                            insertion_idx += 1
                    except IndexError:
                        pass

                    unfinished_paths.insert(insertion_idx, new_path)
            else:
                if optimal_path is None or old_path["length"] < optimal_path["length"]:
                    optimal_path = old_path

        ros.loginfo(f"Shortest path length: {optimal_path['length']}")
        if optimal_path["length"] > np.pi:
            return None
        else:
            path = np.stack([node["pose"] for node in optimal_path["nodes"]])

            joint_pose = path[0]

            return path
