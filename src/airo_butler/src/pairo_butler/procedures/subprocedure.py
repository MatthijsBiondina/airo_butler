from abc import ABC, abstractmethod

from munch import Munch
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

    @abstractmethod
    def run():
        pass
