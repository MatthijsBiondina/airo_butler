import base64
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rospkg
from pairo_butler.utils.pods import POD
from pairo_butler.utils.ros_message_collector import ROSMessageCollector
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.tools import makedirs, pbar, pyout
import rospy as ros
from rostopic import get_topic_class
from airo_butler.srv import Reset
from airo_butler.msg import PODMessage


class DataCollector:
    QUEUE_SIZE = 2
    RATE = 30
    BLACKLIST = ["/rosout", "/rosout_agg"]
    WHITELIST = [
        "/rs2_topic",
        "/ur3_state_sophie",
    ]

    def __init__(
        self,
        name: str = "data_collector",
        root: str = "/media/matt/Expansion/Datasets/towels",
    ):
        self.node_name: str = name
        self.root: Path = Path(root)
        assert self.root.exists()

        self.collector: Optional[ROSMessageCollector] = None

        self.subscribers: List[ros.Subscriber] = []

        # Placeholders

        self.__recording: bool = False
        self.__saving: bool = False
        self.packages: Dict[str, List[POD]] = {}
        self.reset()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        for topic in self.WHITELIST:
            self.subscribers.append(
                ros.Subscriber(
                    topic, PODMessage, self.__sub_callback, callback_args=topic
                )
            )

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

    def __sub_callback(self, msg, topic):
        if self.__recording:
            self.packages[topic].append(msg)

    def reset(self):
        for topic in self.WHITELIST:
            self.packages[topic] = []

    def run(self):
        while not ros.is_shutdown():
            if self.__saving:
                ros.loginfo("Saving!")
                self.__compress_images_to_mp4_and_store_data()
                self.__saving = False

            self.rate.sleep()

    @staticmethod
    def start_recording():
        invoke_service("start_recording_service")

    def __start_recording_callback(self, _):
        while self.__saving:
            ros.Time.sleep(1 / self.RATE)

        self.reset()
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
        self.__saving = True
        return True

    def __compress_images_to_mp4_and_store_data(self):
        rs2_pods = []
        for msg in pbar(self.packages["/rs2_topic"], desc="Deserializing rs2 messages"):
            rs2_pods.append(pickle.loads(msg.data))
        tcp_pods = []
        for msg in pbar(
            self.packages["/ur3_state_sophie"], desc="Deserializing tcp messages"
        ):
            tcp_pods.append(pickle.loads(msg.data))

        rs2_time = np.array(
            [pod.timestamp.secs + 1e-9 * pod.timestamp.nsecs for pod in rs2_pods]
        )
        tcp_time = np.array(
            [pod.timestamp.secs + 1e-9 * pod.timestamp.nsecs for pod in tcp_pods]
        )

        simultaneous_idx = np.argmin(
            np.absolute(rs2_time[:, None] - tcp_time[None, :]), axis=1
        )

        file_index = 0
        while os.path.exists(self.root / str(file_index).zfill(6)):
            file_index += 1
        save_dir = self.root / str(file_index).zfill(6)

        makedirs(save_dir)

        data = {
            "state_sophie": [],
            "rs2_intrinsics": rs2_pods[0].intrinsics_matrix.tolist(),
        }

        tmp_dir = Path(rospkg.RosPack().get_path("airo_butler")) / "res" / "tmp"
        shutil.rmtree(tmp_dir)
        makedirs(tmp_dir)

        for ii, pod in pbar(
            enumerate(rs2_pods), desc="Compressing Images", total=len(rs2_pods)
        ):
            pod.image.save(tmp_dir / (str(ii).zfill(4) + ".png"), "PNG")
            state = tcp_pods[simultaneous_idx[ii]]
            data["state_sophie"].append(
                {
                    "gripper_width": state.gripper_width,
                    "joint_configuration": state.joint_configuration.tolist(),
                    "tcp_pose": state.tcp_pose.tolist(),
                    "timestamp": {
                        "secs": state.timestamp.secs,
                        "nsecs": state.timestamp.nsecs,
                    },
                }
            )
        os.system(
            f"ffmpeg -framerate 24 -i {tmp_dir}/%04d.png "
            f"-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
            f"{save_dir}/video.mp4"
        )

        with open(save_dir / "state.json", "w+") as f:
            json.dump(data, f, indent=2)

        self.reset()
        return True


def main():
    node = DataCollector()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
