import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import PIL
import cv2
import numpy as np
import rospkg
import yaml
from pairo_butler.labelling.determine_visibility import VisibilityChecker
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, pyout
import rospy as ros


class COCODatasetBuilder:
    def __init__(self, name="coco_builder"):
        self.node_name = name

        config_path: Path = Path(__file__).parent / "labelling_config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

    def start_ros(self) -> None:
        ros.init_node(self.node_name, log_level=ros.INFO)

        ros.loginfo(f"{self.node_name}: OK!")
