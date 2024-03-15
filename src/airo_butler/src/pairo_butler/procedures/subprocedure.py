from abc import ABC, abstractmethod
import pickle
from typing import Optional
import rospy as ros
from airo_butler.msg import PODMessage
from munch import Munch
import numpy as np
from pairo_butler.motion_planning.ompl_client import OMPLClient
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient


class Subprocedure(ABC):
    QUEUE_SIZE = 2
    PUBLISH_RATE = 60

    def __init__(self, *args, **kwargs):
        self.sophie: UR5eClient = kwargs["sophie"]
        self.wilson: UR5eClient = kwargs["wilson"]
        self.ompl: OMPLClient = kwargs["ompl"]
        self.config: Munch = kwargs["config"]

        self.towel_sub: ros.Subscriber = ros.Subscriber(
            "/towel_top", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.towel_bot: ros.Subscriber = ros.Subscriber(
            "/towel_bot",
            PODMessage,
            self.__sub_bot_callback,
            queue_size=self.QUEUE_SIZE,
        )
        self.towel_top: Optional[np.ndarray] = None
        self.towel_bot: Optional[np.ndarray] = None

    @abstractmethod
    def run():
        pass

    def __sub_callback(self, msg):
        pod = pickle.loads(msg.data)
        self.towel_top = np.array([pod.x, pod.y, pod.z])

    def __sub_bot_callback(self, msg):
        pod = pickle.loads(msg.data)
        self.towel_bot = np.array([pod.x, pod.y, pod.z])

    def towel_on_table(self):
        return not (self.towel_bot[2] > 0.05 and self.towel_bot[2] < 1.0)
