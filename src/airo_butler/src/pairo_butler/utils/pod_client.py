import pickle
import sys
from typing import List, Optional, Type
from pairo_butler.utils.pods import POD
import rospy as ros
from airo_butler.msg import PODMessage


class PODClient:
    QUEUE_SIZE = 2
    RATE = 60

    def __init__(self, topic: str, pod_type: Type, timeout: int = 5):
        self.timeout = None if timeout <= 0 else ros.Duration(timeout)
        self.topic = topic
        self.msg_type = pod_type
        self.rate = ros.Rate(self.RATE)
        self.subscriber: ros.Subscriber = ros.Subscriber(
            topic, PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.__pod: Optional[POD] = None
        self.__timestamps: List[ros.Time] = []

    @property
    def pod(self) -> POD:
        t0 = ros.Time.now()
        while self.__pod is None and (
            self.timeout is None or ros.Time.now() < t0 + self.timeout
        ):
            self.rate.sleep()

        if self.__pod is None:
            ros.logerr(f"No POD received on {self.topic}. Is it being published?")
            ros.signal_shutdown(f"Did not receive POD on {self.topic}.")
            sys.exit(0)

        return self.__pod

    @property
    def fps(self) -> int:
        return len(self.__timestamps)

    @property
    def latency(self) -> int:
        try:
            latency = ros.Time.now() - self.__timestamps[-1]
            return int(latency.to_sec() * 1000)
        except IndexError:
            return -1

    def __sub_callback(self, msg: PODMessage):
        pod: self.msg_type = pickle.loads(msg.data)
        self.__pod = pod
        self.__timestamps.append(pod.timestamp)
        while pod.timestamp - ros.Duration(secs=1) > self.__timestamps[0]:
            self.__timestamps.pop(0)
