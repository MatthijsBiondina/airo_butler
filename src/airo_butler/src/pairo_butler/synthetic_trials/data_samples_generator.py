from multiprocessing import Process
from threading import Lock
import time
from typing import Dict
from pairo_butler.utils.pods import ImagePOD, URStatePOD, publish_pod
from pairo_butler.labelling.labelling_utils import LabellingUtils
from pairo_butler.utils.tools import listdir, load_config, pyout
import rospy as ros
from airo_butler.msg import PODMessage
from airo_butler.srv import PODService, Reset


NAME = "synthetic_data_generator"


class SyntheticDataGenerator:
    QUEUE_SIZE = 2

    def __init__(self):
        self.node_name: str = NAME
        self.config = load_config()
        self.lock = Lock()

        self.rs2_publisher: ros.Publisher
        self.sophie_publisher: ros.Publisher

        self.reset_service: ros.Service
        self.dummy_rs2_reset_service: ros.Service

        self.trials = listdir(self.config.data_root)
        self.trial_idx = -1
        self.buffer = []
        self.__load_next_trial_in_buffer()
        self.reload_timestamp: ros.Time

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.config.rate)

        self.reset_service = ros.Service(
            f"reset_{self.node_name}", Reset, self.__reset_service_callback
        )
        self.rs2_publisher = ros.Publisher(
            "/rs2_topic", PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.sophie_publisher = ros.Publisher(
            "/ur5e_sophie", PODMessage, queue_size=self.QUEUE_SIZE
        )

        self.dummy_services = self.__init_dummy_services()

        self.reload_timestamp = ros.Time.now()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            with self.lock:
                if len(self.buffer):
                    sample = self.buffer.pop(0)

                    self.__publish_sophie_state(sample["sophie"])
                    self.__publish_rs2(sample["rs2"])

            self.rate.sleep()

    @staticmethod
    def reset():
        """
        Static method to invoke the reset service of a SyntheticDataGenerator node.
        :param node_name: The name of the node for which the reset service is to be invoked.
        """
        service_name = f"reset_{NAME}"
        try:
            ros.wait_for_service(service_name, timeout=5)
            reset_service = ros.ServiceProxy(service_name, Reset)
            response = reset_service()
            pyout(f"Reset service invoked successfully: {response}")
        except ros.ROSException as e:
            pyout(f"Failed to connect to the reset service: {e}")
        except ros.ServiceException as e:
            pyout(f"Service call failed: {e}")

    # Example usage within the same script or module:

    def __init_dummy_services(self) -> Dict[str, ros.Service]:
        services = {
            "reset_realsense_service": ros.Service(
                "reset_realsense_service", Reset, lambda *args: True
            ),
            "move_to_joint_configuration": ros.Service(
                "move_to_joint_configuration",
                PODService,
                lambda *args: True,
            ),
            "execute_trajectory": ros.Service(
                "execute_trajectory",
                PODService,
                lambda *args: True,
            ),
            "move_gripper": ros.Service(
                "move_gripper",
                PODService,
                lambda *args: True,
            ),
            "interrupt": ros.Service(
                "interrupt",
                PODService,
                lambda *args: True,
            ),
        }

        return services

    def __reset_service_callback(self, req):
        self.__load_next_trial_in_buffer()
        self.reload_timestamp = ros.Time.now()
        return True

    def __load_next_trial_in_buffer(self):
        with self.lock:
            while len(self.buffer):
                self.buffer.pop(0)

            while True:
                try:
                    self.trial_idx = (self.trial_idx + 1) % len(self.trials)
                    data, valid = LabellingUtils.load_trial_data(
                        self.trials[self.trial_idx]
                    )
                    if valid:
                        break
                except Exception as e:
                    ros.logwarn(f"Unexpected exception: {e}")

            for state_sophie, rs2_frame in zip(data["state_sophie"], data["frames"]):
                self.buffer.append(
                    {
                        "sophie": state_sophie,
                        "rs2": {
                            "frame": rs2_frame,
                            "intrinsics": data["rs2_intrinsics"],
                        },
                    }
                )

        self.reset_service: ros.Service
        self.dummy_rs2_reset_service: ros.Service

        self.trials = listdir(self.config.data_root)
        self.trial_idx = -1
        self.buffer = []
        self.__load_next_trial_in_buffer()
        self.reload_timestamp: ros.Time

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.config.rate)

        self.reset_service = ros.Service(
            f"reset_{self.node_name}", Reset, self.__reset_service_callback
        )
        self.rs2_publisher = ros.Publisher(
            "/rs2_topic", PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.sophie_publisher = ros.Publisher(
            "/ur5e_sophie", PODMessage, queue_size=self.QUEUE_SIZE
        )

        self.dummy_services = self.__init_dummy_services()

        self.reload_timestamp = ros.Time.now()

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            with self.lock:
                if len(self.buffer):
                    sample = self.buffer.pop(0)

                    self.__publish_sophie_state(sample["sophie"])
                    self.__publish_rs2(sample["rs2"])

            self.rate.sleep()

    @staticmethod
    def reset():
        """
        Static method to invoke the reset service of a SyntheticDataGenerator node.
        :param node_name: The name of the node for which the reset service is to be invoked.
        """
        service_name = f"reset_{NAME}"
        try:
            ros.wait_for_service(service_name, timeout=5)
            reset_service = ros.ServiceProxy(service_name, Reset)
            response = reset_service()
            pyout(f"Reset service invoked successfully: {response}")
        except ros.ROSException as e:
            pyout(f"Failed to connect to the reset service: {e}")
        except ros.ServiceException as e:
            pyout(f"Service call failed: {e}")

    # Example usage within the same script or module:

    def __init_dummy_services(self) -> Dict[str, ros.Service]:
        services = {
            "reset_realsense_service": ros.Service(
                "reset_realsense_service", Reset, lambda *args: True
            ),
            "move_to_joint_configuration": ros.Service(
                "move_to_joint_configuration",
                PODService,
                lambda *args: True,
            ),
            "execute_trajectory": ros.Service(
                "execute_trajectory",
                PODService,
                lambda *args: True,
            ),
            "move_gripper": ros.Service(
                "move_gripper",
                PODService,
                lambda *args: True,
            ),
            "interrupt": ros.Service(
                "interrupt",
                PODService,
                lambda *args: True,
            ),
        }

        return services

    def __reset_service_callback(self, req):
        self.__load_next_trial_in_buffer()
        self.reload_timestamp = ros.Time.now()
        return True

    def __load_next_trial_in_buffer(self):
        with self.lock:
            while len(self.buffer):
                self.buffer.pop(0)

            while True:
                try:
                    self.trial_idx = (self.trial_idx + 1) % len(self.trials)
                    data, valid = LabellingUtils.load_trial_data(
                        self.trials[self.trial_idx]
                    )
                    if valid:
                        break
                except Exception as e:
                    ros.logwarn(f"Unexpected exception: {e}")

            for state_sophie, rs2_frame in zip(data["state_sophie"], data["frames"]):
                self.buffer.append(
                    {
                        "sophie": state_sophie,
                        "rs2": {
                            "frame": rs2_frame,
                            "intrinsics": data["rs2_intrinsics"],
                        },
                    }
                )

    def __publish_sophie_state(self, state):
        pod = URStatePOD(
            tcp_pose=state["tcp_pose"],
            joint_configuration=state["joint_configuration"],
            gripper_width=state["gripper_width"],
            timestamp=ros.Time.now(),
            arm_name="sophie",
        )

        publish_pod(self.sophie_publisher, pod)

    def __publish_rs2(self, rs2_package):
        pod = ImagePOD(
            color_frame=rs2_package["frame"],
            depth_frame=None,
            image=rs2_package["frame"],
            intrinsics_matrix=rs2_package["intrinsics"],
            timestamp=ros.Time.now(),
        )
        publish_pod(self.rs2_publisher, pod)


def main():
    node = SyntheticDataGenerator()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
