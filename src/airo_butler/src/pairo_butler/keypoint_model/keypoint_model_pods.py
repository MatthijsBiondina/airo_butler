from pathlib import Path
from typing import List
from pairo_butler.orientation_model.orientation_pods import Coord


class KeypointSampleMetaDataPOD:
    __slots__ = ["path", "keypoints"]

    def __init__(self, path: Path, keypoints: List[Coord]):
        self.path: Path = path
        self.keypoints: List[Coord] = keypoints
