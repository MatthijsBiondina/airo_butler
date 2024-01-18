from typing import Optional
import numpy as np
import rospy as ros
from airo_butler.msg import NPMessage
from std_msgs.msg import Header


def serialize_numpy_array(np_array: np.ndarray) -> NPMessage:
    msg = NPMessage()
    msg.data = np_array.tobytes()
    msg.shape = np_array.shape
    msg.dtype = str(np_array.dtype)
    return msg


def deserialize_np_message(msg: NPMessage):
    shape = tuple(msg.shape)
    dtype = np.dtype(msg.dtype)
    np_array = np.frombuffer(msg.data, dtype=dtype)
    np_array = np_array.reshape(shape)
    return np_array, msg.header.timestamp


def publish_np_array(
    publisher: ros.Publisher, np_array: np.ndarray, timestamp: Optional[ros.Time] = None
):
    msg = serialize_numpy_array(np_array=np_array)
    msg.header.stamp = ros.Time.now() if timestamp is None else timestamp
    publisher.publish(msg)
