#!/usr/bin/env python3
import pickle

import rospy as ros
from airo_butler.utils.tools import pyout

from airo_butler.main import MyPOD
from airo_butler.srv import MoveToJointConfiguration


class URs_server:
    def __init__(self):
        print("Starting URs_server... ", end="")
        self.service = ros.Service("move_to_joint_configuration", MoveToJointConfiguration, self.move_to_joint_configuration)
        print("Ready!")

    def move_to_joint_configuration(self, req):
        try:
            pod: MyPOD = pickle.loads(req.pod)

            pyout()
        except Exception as e:
            pyout("foo")

        return True


def main():
    ros.init_node("URs_server", log_level=ros.INFO)
    server = URs_server()
    ros.spin()


if __name__ == "__main__":
    main()
