from datetime import datetime
from multiprocessing import Process
from os import listdir, makedirs
import os
import pickle
import shutil
import sys
import time
from typing import List
import cv2
import PIL
import numpy as np
from pairo_butler.utils.ros_helper_functions import invoke_service
from pairo_butler.utils.pods import ImagePOD
from pairo_butler.utils.tools import load_config, pbar, pyout
import rospy as ros
from airo_butler.msg import PODMessage
from airo_butler.srv import Reset


def invoke():
    time.sleep(1)
    RS2Recorder.start()
    time.sleep(5)
    RS2Recorder.save()


class RS2Recorder:
    RATE = 120
    FPS = 30

    def __init__(self, name: str = "rs2_recorder"):
        self.node_name = name
        self.rate: ros.Rate
        self.config = load_config()

        self.subscriber: ros.Subscriber

        self.frames: List[np.ndarray] = []

        shutil.rmtree(self.config.tmp_dir, ignore_errors=True)
        makedirs(self.config.tmp_dir)
        makedirs(self.config.save_dir, exist_ok=True)

        self.paused: bool = True
        self.t_start: ros.Time
        self.t_end: ros.Time

        self.video_writer = None

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.subscriber = ros.Subscriber(
            "/recorder_rs2", PODMessage, self.__sub_callback, queue_size=2
        )

        self.services = [
            ros.Service("start_rs2_recorder_service", Reset, self.__start),
            ros.Service("stop_rs2_recorder_service", Reset, self.__stop),
            ros.Service("save_rs2_recorder_service", Reset, self.__save),
        ]

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            if len(self.frames) and self.video_writer is not None:
                try:
                    img = self.frames.pop(0)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(img)
                except Exception as e:
                    ros.logwarn(f"Unexpected exception: {e}")
                    pass

            self.rate.sleep()

    @staticmethod
    def start():
        return invoke_service("start_rs2_recorder_service")

    def __start(self, req):
        try:
            self.video_writer.release()
        except AttributeError:
            pass

        shutil.rmtree(self.config.tmp_dir, ignore_errors=True)
        makedirs(self.config.tmp_dir)

        height, width = 720, 1280
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        now = datetime.fromtimestamp(ros.Time.now().to_sec())
        fname = f"{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.filename = fname

        fps = 24
        self.video_writer = cv2.VideoWriter(
            f"{self.config.tmp_dir}/{fname}", fourcc, fps, (width, height)
        )

        self.frames = []
        self.paused = False
        self.t_start = ros.Time.now()
        self.t_end = ros.Time.now()

        return True

    @staticmethod
    def stop():
        return invoke_service("stop_rs2_recorder_service")

    def __stop(self, req):
        self.paused = True

        return True

    @staticmethod
    def save():
        return invoke_service("save_rs2_recorder_service")

    def __save(self, req):
        try:
            self.paused = True

            while len(self.frames):
                ros.loginfo(f"{len(self.frames)} remaining...")
                ros.sleep(1)

            self.video_writer.release()
            self.video_writer = None

            shutil.move(
                f"{self.config.tmp_dir}/{self.filename}",
                f"{self.config.save_dir}/{self.filename}",
            )

            shutil.rmtree(self.config.tmp_dir, ignore_errors=True)
            makedirs(self.config.tmp_dir)

            return True
        except Exception as e:
            ros.logerr(f"Unexpected exception: {e}")
            return False

    def __sub_callback(self, msg):
        if not self.paused:
            pod: ImagePOD = pickle.loads(msg.data)
            self.frames.append(pod.color_frame)
            self.t_end = ros.Time.now()


def main():
    node = RS2Recorder()
    node.start_ros()
    # Process(target=invoke, daemon=True).start()
    node.run()


if __name__ == "__main__":
    main()
