import base64
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Dict, List, Optional
from PIL import Image
import cv2
import numpy as np
import rospkg
from pairo_butler.data.timesync import TimeSync
from pairo_butler.utils.pods import POD, BooleanPOD, StringPOD, make_pod_request
from pairo_butler.utils.ros_message_collector import ROSMessageCollector
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.tools import load_config, makedirs, pbar, pyout
import rospy as ros
from rostopic import get_topic_class
from airo_butler.srv import Reset, PODService
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
        self.__path: Optional[Path] = None

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
            "save_recording_service", PODService, self.__save_recording_callback
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
                self.__store_data()
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
    def save_recording(path):
        service = ros.ServiceProxy("save_recording_service", PODService)
        pod = StringPOD(timestamp=ros.Time.now(), string=path)
        make_pod_request(service, pod, PODService)

    def __save_recording_callback(self, msg):
        pod: StringPOD = pickle.loads(msg.pod)
        self.__path = Path(pod.string)
        self.__saving = True
        return True

    def __store_data(self):
        root = self.__path
        shutil.rmtree(root, ignore_errors=True)
        makedirs(root / "color")
        makedirs(root / "depth")

        data = {
            "state_sophie": [],
            "rs2_intrinsics": self.packages["/rs2_topic"][0].intrinsics_matrix.tolist(),
        }
        color = np.empty((len(self.packages["/rs2_topic"]), 720, 720, 3))
        depth = np.empty((len(self.packages["/rs2_topic"]), 720, 720))

        for ii, (pod, state) in pbar(
            enumerate(zip(self.packages["/rs2_topic"], self.packages["/ur5e_sophie"])),
            desc="Saving...",
            total=len(self.packages["/rs2_topic"]),
        ):
            img_name = str(ii).zfill(3) + ".jpg"

            cv2.imwrite(
                (root / "color" / img_name).as_posix(),
                np.array(pod.color_frame)[..., ::-1],
            )

            depth_frame = pod.depth_frame.astype(float)
            height, width = depth_frame.shape

            crop = min(width, height)
            left = round((width - crop) / 2)
            top = round((height - crop) / 2)
            right = round((width + crop) / 2)
            bottom = round((height + crop) / 2)
            # Crop the image to the calculated dimensions.
            depth_data = depth_frame[top:bottom, left:right]
            depth_data = cv2.resize(
                depth_data, (720, 720), interpolation=cv2.INTER_NEAREST
            )  # hardcoded from rs2 resolution config

            # image = image.crop((left, top, right, bottom))

            depth_frame = (1 - np.clip(depth_data.astype(float) / 2000.0, 0, 1)) * 255
            depth_frame[depth_frame == 255] = 0
            cv2.imwrite(
                (root / "depth" / img_name).as_posix(), depth_frame.astype(np.uint8)
            )

            # np.save(root / "color" / img_name, np.array(pod.color_frame))
            # np.save(root / "depth" / img_name, np.array(pod.depth_frame))

            color[ii] = np.array(pod.color_frame)
            depth[ii] = depth_data

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

        with open(root / "state.json", "w+") as f:
            json.dump(data, f, indent=2)
        np.save(root / "color.npy", color)
        np.save(root / "depth.npy", depth)
        self.reset()
        return True

    def __frame2pillow(self, frame: np.ndarray):
        """
        Converts a frame captured from the RealSense2 camera to a PIL (Python Imaging
        Library) image.

        This private method transforms a RealSense2 frame into a numpy array, then converts
        this array to a PIL image for further processing. The image is first cropped to a
        square format based on the smaller dimension (height or width) and then resized to
        the specified resolution.

        Args:
            frame (pyrealsense2.frame): A frame captured from the RealSense2 camera.

        Returns:
            PIL.Image: The processed image in PIL format, cropped and resized to the specified resolution.
        """
        # Convert the RealSense2 frame to a numpy array.
        if isinstance(frame, np.ndarray):
            img_array = frame.astype(np.uint8)
        else:
            img_array = np.asanyarray(frame.get_data()).astype(np.uint8)
        # Create a PIL image from the numpy array.
        image = Image.fromarray(img_array)

        # Calculate the dimensions for cropping the image to a square.
        crop = min(image.width, image.height)
        left = round((image.width - crop) / 2)
        top = round((image.height - crop) / 2)
        right = round((image.width + crop) / 2)
        bottom = round((image.height + crop) / 2)
        # Crop the image to the calculated dimensions.
        image = image.crop((left, top, right, bottom))

        # Resize the cropped image to the desired resolution.
        image = image.resize((720, 720))

        return image  # Return the processed PIL image.


def main():
    node = DataCollector()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
