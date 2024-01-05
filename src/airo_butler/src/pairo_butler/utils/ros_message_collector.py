import pickle
from typing import List, Optional
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.tools import pyout
from pairo_butler.utils.pods import *


class ROSMessageCollector:
    """ """

    QUEUE_SIZE: int = 2
    BUFFER_SIZE: int = 100

    def __init__(self, exact: List[str], approximate: List[str] = []) -> None:
        self.exact_sub_names: List[str] = exact
        self.aprox_sub_names: List[str] = approximate

        # Buffer contains list of tuples: (timestamp, pod)
        self.buffer = {
            **{topic: [] for topic in exact},
            **{topic: [] for topic in approximate},
        }

        self.callbacks = {
            topic: (lambda msg: self.__sub_callback(msg, topic))
            for topic in exact + approximate
        }
        self.subscribers: List[ros.Subscriber] = []
        for topic in exact + approximate:
            self.subscribers.append(
                ros.Subscriber(
                    topic, PODMessage, self.callbacks[topic], queue_size=self.QUEUE_SIZE
                )
            )

    def __sub_callback(self, msg: PODMessage, topic: str):
        pod: POD = pickle.loads(msg.data)

        try:
            timestamp = pod.timestamp
        except AttributeError:
            timestamp = msg.header.stamp

        ii = 0
        while ii < len(self.buffer[topic]) and timestamp < self.buffer[topic][ii][0]:
            ii += 1

        pyout()
