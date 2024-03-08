import threading
from typing import Dict, Optional

from munch import Munch
from pairo_butler.utils.tools import load_config
from pairo_butler.utils.pods import URStatePOD, publish_pod
import rospy as ros
from airo_robots.manipulators import URrtde
from airo_robots.grippers import Robotiq2F85
from airo_butler.msg import PODMessage
from airo_butler.srv import PODService, PODServiceResponse


class UR5e_server:
    PUBLISH_RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "ur5e_server"):
        self.config: Munch = load_config()

        self.sophie = URrtde(self.config.ip_sophie, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(self.config.ip_sophie)
        self.wilson = URrtde(self.config.ip_wilson, URrtde.UR3E_CONFIG)
        self.wilson.gripper = Robotiq2F85(self.config.ip_wilson)

        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.sophie_publisher_name: str = "/ur5e_sophie"
        self.wilson_publisher_name: str = "/ur5e_wilson"

        # Placeholders for publishers
        self.sophie_publisher: Optional[ros.Publisher] = None
        self.wilson_publisher: Optional[ros.Publisher] = None

        # Initialize services
        self.services: Dict[str, ros.Service] = self.__initialize_services()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.sophie_publisher = ros.Publisher(
            self.sophie_publisher_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        self.wilson_publisher = ros.Publisher(
            self.wilson_publisher_name, PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            timestamp = ros.Time.now()
            pod_sophie = URStatePOD(
                tcp_pose=self.sophie.get_tcp_pose(),
                joint_configuration=self.sophie.get_joint_configuration(),
                gripper_width=self.sophie.gripper.get_current_width(),
                timestamp=timestamp,
                arm_name="sophie",
            )
            pod_wilson = URStatePOD(
                tcp_pose=self.wilson.get_tcp_pose(),
                joint_configuration=self.wilson.get_joint_configuration(),
                gripper_width=self.wilson.gripper.get_current_width(),
                timestamp=timestamp,
                arm_name="wilson",
            )

            publish_pod(self.sophie_publisher, pod_sophie)
            publish_pod(self.wilson_publisher, pod_wilson)

            self.rate.sleep()

    # PRIVATE METHODS
    def __initialize_services(self) -> Dict[str, ros.Service]:
        services = {
            "execute_single_arm_trajectory": ros.Service(
                "execute_single_arm_trajectory",
                PODService,
                self.__execute_single_arm_trajectory,
            ),
            "execute_dual_arm_trajectory": ros.Service(
                "execute_dual_arm_trajectory",
                PODService,
                self.__execute_dual_arm_trajectory,
            ),
            "move_gripper": ros.Service(
                "move_gripper",
                PODService,
                self.__move_gripper,
            ),
            "interrupt_trajectory": ros.Service(
                "interrupt_trajectory",
                PODService,
                self.__interrupt_trajectory,
            ),
        }

    def __execute_single_arm_trajectory(self, req):
        raise NotImplementedError

    def __execute_dual_arm_trajectory(self, req):
        raise NotImplementedError

    def __move_gripper(self, req):
        raise NotImplementedError

    def __interrupt_trajectory(self, req):
        raise NotImplementedError


def main():
    node = UR5e_server()
    node.start_ros()
    threading.Thread(target=node.run, daemon=True).start()
    ros.spin()


if __name__ == "__main__":
    main()
