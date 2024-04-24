import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pairo_butler.utils.pods import POD
from pairo_butler.utils.tools import pyout
import rospy as ros
from airo_butler.msg import PODMessage


class TimeSync:
    QUEUE_SIZE: int = 2
    BUFFER_SIZE: int = 64
    RATE: int = 30

    def __init__(self, ankor_topic: str, unsynced_topics: List[str]):
        self.rate = ros.Rate(self.RATE)
        self.ankor_topic = ankor_topic
        self.unsynced_topics = unsynced_topics

        self.buffer: Dict[str, List[Dict[str, Union[ros.Time, POD]]]] = {
            **{ankor_topic: []},
            **{topic: [] for topic in unsynced_topics},
        }

        self.callbacks = {
            topic: (lambda msg, top_=topic: self.__sub_callback(msg, top_))
            for topic in self.buffer.keys()
        }

        self.subscribers: List[ros.Subscriber] = []
        for topic in self.buffer.keys():
            self.subscribers.append(
                ros.Subscriber(
                    topic, PODMessage, self.callbacks[topic], queue_size=self.QUEUE_SIZE
                )
            )

        self.timestamp: ros.Time = ros.Time.now()

    def __sub_callback(self, msg: PODMessage, topic: str):
        pod: POD = pickle.loads(msg.data)

        try:
            timestamp = pod.timestamp
        except AttributeError:
            timestamp = msg.header.stamp

        ii = 0
        while (
            ii < len(self.buffer[topic])
            and timestamp < self.buffer[topic][ii]["timestamp"]
        ):
            ii += 1
        self.buffer[topic].insert(ii, {"timestamp": timestamp, "pod": pod})

        if len(self.buffer[topic]) > self.BUFFER_SIZE:
            self.buffer[topic] = self.buffer[topic][: self.BUFFER_SIZE]

    def next(self, timeout: Optional[float] = None):
        t_start = ros.Time.now()

        while not ros.is_shutdown():
            TimeSync.check_timeout(t_start, timeout)
            if (
                len(self.buffer[self.ankor_topic]) == 0
                or self.buffer[self.ankor_topic][0]["timestamp"] <= self.timestamp
            ):
                self.rate.sleep()
            else:
                break

        ankor_package = self.buffer[self.ankor_topic][0]
        ankor_timestamp = ankor_package["timestamp"]
        self.timestamp = ankor_timestamp

        synchronized_packages = {self.ankor_topic: ankor_package}

        for topic in self.unsynced_topics:
            while not ros.is_shutdown():
                TimeSync.check_timeout(t_start, timeout)
                if (
                    len(self.buffer[topic]) == 0
                    or self.buffer[topic][0]["timestamp"] < ankor_timestamp
                ):
                    self.rate.sleep()
                else:
                    break

            dt_min = ros.Duration(secs=sys.maxsize)
            package_closest: Optional[POD] = None
            for package in self.buffer[topic]:
                dt = abs(ankor_timestamp - ankor_timestamp)
                if dt < dt_min:
                    dt_min = dt
                    package_closest = package
                else:
                    break
            synchronized_packages[topic] = package_closest

        return synchronized_packages, ankor_timestamp

    @staticmethod
    def check_timeout(t_start: ros.Time, timeout: Optional[float]):
        if timeout is None:
            return

        elapsed_time = ros.Time.now() - t_start
        timeout_duration = ros.Duration(
            secs=int(timeout), nsecs=int((timeout - int(timeout)) * 1e9)
        )
        if elapsed_time > timeout_duration:
            raise TimeoutError("Operation timed out")
