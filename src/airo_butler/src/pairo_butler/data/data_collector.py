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
from pairo_butler.data.timesync import TimeSync
from pairo_butler.utils.pods import POD
from pairo_butler.utils.ros_message_collector import ROSMessageCollector
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.tools import load_config, makedirs, pbar, pyout
import rospy as ros
from rostopic import get_topic_class
from airo_butler.srv import Reset
from airo_butler.msg import PODMessage


class DataCollector:
    QUEUE_SIZE = 2
    RATE = 30

    def __init__(
        self,
        name: str = "data_collector",
    ):
        self.config = load_config()

        self.node_name: str = name
        self.root: Path = Path(self.config.savedir)
        makedirs(self.root)

        self.collector: TimeSync
        self.topics: List[str] = [self.config.ankor_topic] + self.config.unsynced_topics

        self.__recording: bool = False
        self.__saving: bool = False

        self.packages: Dict[str, List[POD]] = {}
        self.reset()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.collector = TimeSync(
            ankor_topic=self.config.ankor_topic,
            unsynced_topics=self.config.unsynced_topics,
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

    def reset(self):
        for topic in self.topics:
            self.packages[topic] = []

    def run(self):
        while not ros.is_shutdown():
            if self.__recording:
                chunk, _ = self.collector.next()
                for topic, topic_data in chunk.items():
                    try:
                        self.packages[topic].append(topic_data["pod"])
                    except KeyError:
                        self.packages[topic] = [topic_data["pod"]]

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
            self.rate.sleep()

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

        file_index = 0
        while os.path.exists(self.root / str(file_index).zfill(6)):
            file_index += 1
        save_dir = self.root / str(file_index).zfill(6)

        makedirs(save_dir)

        data = {
            "state_sophie": [],
            "rs2_intrinsics": self.packages["/rs2_topic"][0].intrinsics_matrix.tolist(),
        }

        res_dir = Path(rospkg.RosPack().get_path("airo_butler")) / "res"
        tmp_dir = res_dir / "tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        makedirs(tmp_dir)

        for ii, (pod, state) in pbar(
            enumerate(zip(self.packages["/rs2_topic"], self.packages["/ur5e_sophie"])),
            desc="Compressing Images",
            total=len(self.packages["/rs2_topic"]),
        ):
            pod.image.save(tmp_dir / (str(ii).zfill(4) + ".png"), "PNG")
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

        cmd = (
            f"ffmpeg -framerate 24 -i {tmp_dir}/%04d.png "
            f"-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
            f"{res_dir}/video.mp4"
        )

        os.system(cmd)
        shutil.move(f"{res_dir}/video.mp4", f"{save_dir}/video.mp4")

        with open(save_dir / "state.json", "w+") as f:
            json.dump(data, f, indent=2)

        self.reset()
        # shutil.rmtree(tmp_dir, ignore_errors=True)
        return True


def main():
    node = DataCollector()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
