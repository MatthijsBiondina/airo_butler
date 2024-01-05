import PIL
from PIL import Image

import rospy as ros
import pickle
import sys
from typing import Optional, Type, Union
import numpy as np
from airo_butler.msg import PODMessage
from airo_butler.srv import PODServiceRequest
from pairo_butler.utils.tools import pyout


class POD:
    """
    Marker base class for all POD (Plain Old Data) types.
    Used as a superclass for data classes that are used to store and transmit simple data.
    """

    pass


class UR3GripperPOD(POD):
    """
    Represents the data for controlling a UR3 robot's gripper.

    Attributes:
        pose (Union[str, float]): The target pose of the gripper. Can be a float representing the position
                                  or a string ("open" or "close").
        side (str): Indicates which arm ("left" or "right") the gripper is on.
        blocking (bool): If True, the movement will be blocking.
    """

    __slots__ = ["pose", "side", "blocking"]

    def __init__(
        self, pose: Union[str, float], left_or_right_arm: str, blocking: bool = True
    ):
        assert isinstance(pose, float) or pose in ["open", "close"]
        self.pose: Union[str, float] = pose
        assert left_or_right_arm in ["left", "right"]
        self.side: str = left_or_right_arm
        self.blocking = blocking


class UR3PosePOD(POD):
    """
    Represents the pose data for a UR3 robot arm.

    Attributes:
        pose (np.ndarray): Numpy array representing the pose.
        joint_speed (Optional[float]): The speed of the joint movement.
        side (str): Indicates which arm ("left" or "right") the data is for.
        blocking (bool): If True, the movement will be blocking.
    """

    __slots__ = ["pose", "joint_speed", "side", "blocking"]

    def __init__(
        self,
        pose: np.ndarray,
        left_or_right_arm: str,
        joint_speed: Optional[float] = None,
        blocking: bool = True,
    ):
        self.pose: np.ndarray = pose
        assert left_or_right_arm in ["left", "right"]
        self.side = left_or_right_arm
        assert joint_speed is None or joint_speed > 0
        self.joint_speed: Optional[float] = joint_speed
        self.blocking = blocking


class UR3StatePOD(POD):
    __slots__ = ["tcp_pose", "joint_configuration"]

    def __init__(self, tcp_pose: np.ndarray, joint_configuration: np.ndarray):
        self.tcp_pose: np.ndarray = tcp_pose
        self.joint_configuration: np.ndarray = joint_configuration


class BooleanPOD(POD):
    """
    Represents a simple boolean data structure.

    Attributes:
        value (bool): The boolean value stored in this POD.
    """

    __slots__ = ["value"]

    def __init__(self, value: bool):
        self.value: bool = value


class ImagePOD(POD):
    """
    Represents a Pillow image.

    Attributes:
        value (PIL.Image): Pillow image
        timestamp
    """

    __slots__ = ["image", "timestamp"]

    def __init__(self, image: Image, timestamp: ros.Time):
        self.image = image
        self.timestamp = timestamp


class ArrayPOD(POD):
    """
    Represents a numpy array with timestamp.

    Attributes:
        value (np.ndarray): numpy array
        timestamp
    """

    __slots__ = ["array", "timestamp"]

    def __init__(self, array: np.ndarray, timestamp: ros.Time) -> None:
        self.array = array
        self.timestamp = timestamp


def make_pod_request(
    service: ros.ServiceProxy, pod: POD, response_type: Type[POD]
) -> POD:
    """
    Sends a request to a ROS service with a given POD object and waits for a response.

    Parameters:
        service (ros.ServiceProxy): The ROS service to which the request is sent.
        pod (POD): The POD object to be sent as part of the service request.
        response_type (Type[POD]): The expected type of the POD response from the service.

    Returns:
        POD: The response POD object from the service.

    Raises:
        ros.ServiceException: If there's an issue with the service call.
        ros.ROSInterruptException: If the ROS node is interrupted.
    """
    # Serialize the POD object
    msg = pickle.dumps(pod)

    # Create a service request and set the POD field
    service_request: PODServiceRequest = PODServiceRequest()
    service_request.pod = msg

    try:
        # Call the service with the request
        response: response_type = pickle.loads(service(service_request).pod)
        return response
    except ros.ServiceException as e:
        pyout()
    except ros.ROSInterruptException:
        sys.exit(0)


def publish_pod(publisher: ros.Publisher, pod: POD):
    msg = PODMessage()
    msg.data = pickle.dumps(pod)
    msg.header.stamp = ros.Time.now()
    publisher.publish(msg)
