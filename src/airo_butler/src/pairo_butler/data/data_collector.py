import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.tools import pyout
import rospy as ros
from rostopic import get_topic_class
from airo_butler.srv import Reset


class DataCollector:
    QUEUE_SIZE = 2
    BLACKLIST = ["/rosout", "/rosout_agg"]
    WHITELIST = [
        "/rs2_topic",
        "/recorder_topic",
        "/ur3_state_sophie",
        "/ur3_state_wilson",
    ]

    def __init__(
        self,
        name: str = "data_collector",
        root: str = "/media/matt/Expansion/Datasets/towels",
    ):
        self.node_name: str = name
        self.root: Path = Path(root)
        assert self.root.exists()

        self.subscribers: List[ros.Subscriber] = []

        # Placeholders
        self.__recording: bool = False
        self.packages: Dict[str, List[Dict[str, Any]]] = {}

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        for topic, _ in ros.get_published_topics():
            if topic in self.BLACKLIST or topic not in self.WHITELIST:
                continue

            msg_class, _, _ = get_topic_class(topic)
            if msg_class is not None:
                self.subscribers.append(
                    ros.Subscriber(
                        topic,
                        msg_class,
                        self.generic_callback,
                        callback_args=topic,
                        queue_size=self.QUEUE_SIZE,
                    )
                )
                self.packages[topic] = []

        self.start_recording_service = ros.Service(
            "start_recording_service", Reset, self.__start_recording_callback
        )
        self.pause_recording_service = ros.Service(
            "pause_recording_service", Reset, self.__pause_recording_callback
        )
        self.save_recording_service = ros.Service(
            "save_recording_service", Reset, self.__save_recording_callback
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def generic_callback(self, msg, topic):
        if self.__recording:
            self.packages[topic].append(base64.b64encode(msg.data).decode("utf-8"))

    @staticmethod
    def start_recording():
        invoke_service("start_recording_service")

    def __start_recording_callback(self, _):
        for topic in self.packages:
            self.packages[topic] = []
        self.__recording = True
        return True

    @staticmethod
    def pause_recording():
        invoke_service("pause_recording_service")

    def __pause_recording_callback(self, _):
        self.__recording = False
        return True

    @staticmethod
    def save_recording():
        invoke_service("save_recording_service")

    def __save_recording_callback(self, _):
        file_index = 0
        while os.path.exists(self.root / (str(file_index).zfill(6) + ".json")):
            file_index += 1
        file = self.root / (str(file_index).zfill(6) + ".json")

        with open(file, "w+") as f:
            json.dump(self.packages, f, indent=2)

        pyout()


def main():
    node = DataCollector()
    node.start_ros()
    DataCollector.start_recording()
    ros.sleep(1)
    DataCollector.pause_recording()
    ros.sleep(0.5)
    DataCollector.save_recording()

    ros.spin()


if __name__ == "__main__":
    main()
