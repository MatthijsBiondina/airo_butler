#!/usr/bin/env python3
from typing import Optional
import rospy as ros
from airo_butler.utils.tools import pyout
from std_msgs.msg import String

class Talker:
    QUEUE_SIZE: int = 10
    PUBLISH_RATE: int = 10

    def __init__(self, name: str = "talker"):
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None

        # Declare ROS publishers
        self.pub_name: str = "/chatter"

        # Placeholder for publisher
        self.pub: Optional[ros.Publisher] = None

    def start_ros(self):
        """
        Create ros node for this class and initialize subscribers and publishers
        """
        # Create a ROS node with a name for this class
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)

        # Define a callback to stop the program
        ros.on_shutdown(self.stop_node)

        # Init publisher
        self.pub = ros.Publisher(self.pub_name, String, queue_size=self.QUEUE_SIZE)

    def stop_node(self):
        """
        Cleanly close the node
        """
        pass

    def run(self):

        while not ros.is_shutdown():
            hello_str = f"Hello world! {ros.get_time()}"
            pyout(f"Publishing: {hello_str}")
            self.pub.publish(hello_str)
            self.rate.sleep()


def main():
    node = Talker()
    node.start_ros()
    node.run()

if __name__ == "__main__":
    main()
