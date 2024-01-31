from copy import deepcopy
import pickle
import time
from typing import List, Optional
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.tools import pyout, rostime2datetime
from pairo_butler.utils.pods import *


class ROSMessageCollector:
    """ """

    QUEUE_SIZE: int = 2
    BUFFER_SIZE: int = 32
    RATE: int = 30

    def __init__(self, exact: List[str], approximate: List[str] = []) -> None:
        self.exact_topics: List[str] = exact
        self.aprox_topics: List[str] = approximate

        # Buffer contains list of tuples: (timestamp, pod)
        self.buffer = {
            **{topic: [] for topic in exact},
            **{topic: [] for topic in approximate},
        }

        self.callbacks = {
            topic: (lambda msg, t=topic: self.__sub_callback(msg, t))
            for topic in exact + approximate
        }
        self.subscribers: List[ros.Subscriber] = []
        for topic in exact + approximate:
            self.subscribers.append(
                ros.Subscriber(
                    topic, PODMessage, self.callbacks[topic], queue_size=self.QUEUE_SIZE
                )
            )

        self.timestamp = ros.Time.now()

    def __sub_callback(self, msg: PODMessage, topic: str):
        pod: POD = pickle.loads(msg.data)

        try:
            timestamp = pod.timestamp
        except AttributeError:
            timestamp = msg.header.stamp

        ii = 0
        while ii < len(self.buffer[topic]) and timestamp < self.buffer[topic][ii][0]:
            ii += 1

        self.buffer[topic].insert(ii, (timestamp, pod))

        if len(self.buffer[topic]) > self.BUFFER_SIZE:
            self.buffer[topic] = self.buffer[topic][: self.BUFFER_SIZE]

        assert self.buffer[topic][0][0] == timestamp

    def next(self, timeout: Optional[float] = None):
        t0 = time.time()
        while timeout is None or time.time() - t0 < timeout:
            stamp, exact_pods = self.__collect_exact()
            if stamp:
                remaining_time = (
                    None if timeout is None else timeout - (time.time() - t0)
                )
                aprox_pods = self.__collect_approx(stamp, timeout=remaining_time)
                if exact_pods is not None and aprox_pods is not None:
                    self.timestamp = stamp
                    return {"timestamp": stamp, **exact_pods, **aprox_pods}
        raise TimeoutError

    def __collect_exact(self):
        b_exact = {}
        for topic in self.exact_topics:
            b_exact[topic] = {"timestamp": [], "object": []}
            b_exact[topic]["timestamp"] = [tuple_[0] for tuple_ in self.buffer[topic]]
            b_exact[topic]["object"] = [tuple_[1] for tuple_ in self.buffer[topic]]

        for stamp in b_exact[self.exact_topics[0]]["timestamp"]:
            if all(
                stamp in b_exact[topic]["timestamp"] for topic in self.exact_topics[1:]
            ):
                if stamp <= self.timestamp:
                    return None, None
                else:
                    D = {}
                    for topic in b_exact.keys():
                        idx = b_exact[topic]["timestamp"].index(stamp)
                        D[topic] = b_exact[topic]["object"][idx]
                    return stamp, D

        return None, None

    def __collect_approx(self, stamp: ros.Time, timeout=None):
        D = {}
        for topic in self.aprox_topics:
            buf_ = deepcopy(self.buffer[topic])
            t0 = time.time()
            while len(buf_) == 0:
                ros.sleep(1 / self.RATE)

            while len(buf_) == 0 or buf_[0][0] < stamp:
                buf_ = deepcopy(self.buffer[topic])
                timediff = stamp - buf_[0][0]
                ros.loginfo(f"{timediff.secs + 1e-9*timediff.nsecs}")

                if timeout is not None and time.time() - t0 > timeout:
                    raise TimeoutError
                else:
                    ros.sleep(1 / self.RATE)
            stamps = [buf_[idx][0] for idx in range(len(buf_))]
            timedelta = [abs(stamp_ - stamp) for stamp_ in stamps]
            idx = min(range(len(timedelta)), key=lambda i: timedelta[i])
            D[topic] = buf_[idx][1]
        return D
