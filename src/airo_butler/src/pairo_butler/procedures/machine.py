from abc import ABC, abstractmethod
from typing import Any, Dict
import rospy as ros
from pairo_butler.motion_planning.ompl_client import OMPLClient
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.utils.tools import load_config


class Machine(ABC):
    def __init__(self, name: str):
        self.node_name = name
        self.config = load_config()

        self.ompl = OMPLClient
        self.sophie: UR5eClient
        self.wilson: UR5eClient

        self.kwargs: Dict[str, Any]

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.ompl = OMPLClient()
        self.sophie = UR5eClient("sophie")
        self.wilson = UR5eClient("wilson")

        self.kwargs = {
            "sophie": self.sophie,
            "wilson": self.wilson,
            "ompl": self.ompl,
            "config": self.config,
        }

        ros.loginfo(f"{self.node_name}: OK!")

    @abstractmethod
    def run(self):
        raise NotImplemented
