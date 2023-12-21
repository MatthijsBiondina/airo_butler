#!/usr/bin/env python3
import pickle

import rospy as ros
from airo_butler.srv import MoveToJointConfiguration, MoveToJointConfigurationRequest
from airo_butler.utils.tools import pyout


class MyPOD:
    __slots__ = ["message"]

    def __init__(self):
        self.message = "Hello World!"

def move_arm():
    ros.wait_for_service("move_to_joint_configuration")
    try:
        move_to_joint_configuration = ros.ServiceProxy("move_to_joint_configuration", MoveToJointConfiguration)

        pod = MyPOD()
        response = move_to_joint_configuration(pickle.dumps(pod))

        pyout()
    except ros.ServiceException as e:
        pyout()
    except ros.ROSInterruptException:
        pass

if __name__ == "__main__":
    ros.init_node("URs_client")
    move_arm()