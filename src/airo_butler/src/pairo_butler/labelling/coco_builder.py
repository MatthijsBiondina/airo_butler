import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import PIL
import cv2
import numpy as np
import rospkg
import yaml
from pairo_butler.labelling.coco_dataset_structure import COCODatasetStructure
from pairo_butler.labelling.labelling_utils import LabellingUtils
from pairo_butler.labelling.determine_visibility import VisibilityChecker
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, pyout
import rospy as ros


class COCODatasetBuilder:
    SEED = 49

    def __init__(self, name="coco_builder"):
        self.node_name = name

        config_path: Path = Path(__file__).parent / "labelling_config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        random.seed(self.SEED)

    def start_ros(self) -> None:
        ros.init_node(self.node_name, log_level=ros.INFO)

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self) -> None:
        root = Path(self.config["coco_folder"])
        train_trials, eval_trials = self.__make_train_validation_split()
        self.__init_folder_structure()
        self.__build_dataset(train_trials, root / "train")
        self.__build_dataset(eval_trials, root / "validation")

    def __make_train_validation_split(self):
        trials = list(listdir(self.config["folder"]))

        train_set = random.sample(
            trials, int(len(trials) * self.config["train_eval_split"])
        )
        validation_set = [trial for trial in trials if trial not in train_set]

        return train_set, validation_set

    def __init_folder_structure(self):
        root_folder = Path(self.config["coco_folder"])

        root_folder.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(root_folder / "train", ignore_errors=True)
        shutil.rmtree(root_folder / "validation", ignore_errors=True)

        (root_folder / "train" / "images").mkdir(parents=True)
        (root_folder / "validation" / "images").mkdir(parents=True)

    def __build_dataset(self, trials: List[Path], root: Path):
        coco_dataset = COCODatasetStructure()

        for trial in pbar(trials):
            data, valid = LabellingUtils.load_trial_data(trial)
            if not valid:
                continue

            nr_of_frames = len(data["frames"])
            for frame_idx in pbar(range(nr_of_frames), desc=trial.name):
                # resize_image to power of 2
                resized_img = data["frames"][frame_idx].resize((512, 512))
                ratio = 512 / 720
                resized_keypoints = data["keypoints_clean"][frame_idx]
                for kp_idx in range(len(resized_keypoints)):
                    resized_keypoints[kp_idx][0][0] *= ratio
                    resized_keypoints[kp_idx][1][0] *= ratio

                coco_dataset.add_sample(
                    image=resized_img,
                    keypoints=resized_keypoints,
                    orientations=data["keypoints_theta"][frame_idx],
                    root=root,
                )

        with open(root / "annotations.json", "w+") as f:
            json.dump(coco_dataset.dictionary, f, indent=2)


def main():
    node = COCODatasetBuilder()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
