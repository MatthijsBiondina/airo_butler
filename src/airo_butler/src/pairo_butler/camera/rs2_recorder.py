from datetime import datetime
from multiprocessing import Process
from os import listdir, makedirs
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import List
import cv2
from PIL import Image
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
    RS2Recorder.finish()


class RS2Recorder:
    RATE = 120
    FPS = 10

    def __init__(self, name: str = "rs2_recorder"):
        self.node_name = name
        self.rate: ros.Rate
        self.config = load_config()

        self.subscriber: ros.Subscriber

        self.frames: List[np.ndarray] = []

        makedirs(self.config.tmp_dir, exist_ok=True)
        makedirs(self.config.save_dir, exist_ok=True)
        self.folder = None

        self.paused: bool = True
        self.ii = 0
        self.last_frame_timestamp: ros.Time

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)

        self.last_frame_timestamp = ros.Time.now()

        self.subscriber = ros.Subscriber(
            "/recorder_rs2", PODMessage, self.__sub_callback, queue_size=2
        )

        self.services = [
            ros.Service("start_rs2_recorder_service", Reset, self.__start),
            ros.Service("finish_rs2_recorder_service", Reset, self.__finish),
        ]

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            for folder in listdir(self.config.tmp_dir):
                path = Path(self.config.tmp_dir) / folder
                if not path.is_dir():
                    continue

                creation_time = path.stat().st_ctime
                if time.time() - creation_time < 10:
                    continue

                if len(listdir(path)):
                    last_modified_time = max(
                        (path / file).stat().st_ctime for file in listdir(path)
                    )
                    if time.time() - last_modified_time < 10:
                        continue

                    self.__convert_np_to_img(path)
                    self.__compress_image_directory_into_mp4(folder)

                shutil.rmtree(path)

            self.rate.sleep()

    @staticmethod
    def start():
        return invoke_service("start_rs2_recorder_service")

    def __start(self, req):
        now = datetime.fromtimestamp(ros.Time.now().to_sec())

        self.folder = Path(self.config.tmp_dir) / now.strftime("%Y%m%d_%H%M%S")
        makedirs(self.folder)
        self.paused = False

        return True

    @staticmethod
    def finish():
        return invoke_service("finish_rs2_recorder_service")

    def __finish(self, req):
        self.paused = True

        return True

    def __sub_callback(self, msg):
        if not self.paused:
            pod: ImagePOD = pickle.loads(msg.data)
            img_nr = len(listdir(self.folder))
            np.save(self.folder / f"{str(img_nr).zfill(4)}.npy", pod.color_frame)

    def __convert_np_to_img(self, folder: Path):

        np_files = [file for file in listdir(folder) if file.endswith(".npy")]

        for file in pbar(np_files, desc="Compressing images"):
            img_array = np.load(folder / file)
            img = Image.fromarray(img_array)

            img = img.resize((img.width // 2, img.height // 2))
            img.save(folder / file.replace(".npy", ".jpg"))

    def __compress_image_directory_into_mp4(self, folder: Path):
        input_folder = Path(self.config.tmp_dir) / folder
        output_file = Path(self.config.tmp_dir) / f"{folder}.mp4"

        cmd = (
            f"ffmpeg -framerate 30 -i {input_folder}/%04d.jpg "
            f"-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
            f"{output_file}"
        )
        os.system(cmd)

        shutil.move(output_file, Path(self.config.save_dir) / f"{folder}.mp4")


def main():
    node = RS2Recorder()
    node.start_ros()
    # Process(target=invoke, daemon=True).start()
    node.run()


if __name__ == "__main__":
    main()
