from pairo_butler.ur3_arms.ur3_client import SOPHIE_REST, WILSON_REST, UR3Client
import rospy as ros
import sys


class PathCheck:
    def __init__(self, name: str = "PathCheck"):
        self.node_name = name

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        ros.loginfo(f"Python interpreter path: {sys.executable}")


def main():
    node = PathCheck()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    sophie = UR3Client("sophie")
    wilson = UR3Client("wilson")

    sophie.move_to_joint_configuration(SOPHIE_REST)
    wilson.move_to_joint_configuration(WILSON_REST)

    main()
