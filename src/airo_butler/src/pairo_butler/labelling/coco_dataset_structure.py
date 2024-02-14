from pathlib import Path
from typing import List
from PIL import Image
from pairo_butler.utils.tools import pyout


class COCODatasetStructure:
    def __init__(
        self,
        supercategory: str = "cloth",
        name: str = "towel",
        keypoints: List[str] = ["corner0", "corner1"],
    ):
        self.dictionary = {
            "categories": [
                {
                    "supercategory": supercategory,
                    "id": 0,
                    "name": name,
                    "keypoints": keypoints,
                    "skeleton": [],
                }
            ],
            "images": [],
            "annotations": [],
        }

    def add_sample(
        self,
        image: Image.Image,
        keypoints: List[List[float]],
        orientations: List[float],
        root: Path,
        zfill: int = 6,
    ):
        idx = len(self.dictionary["images"])
        self.dictionary["images"].append(
            {
                "id": idx,
                "width": image.width,
                "height": image.height,
                "file_name": f"images/{str(idx).zfill(zfill)}.jpg",
            }
        )
        self.dictionary["annotations"].append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": 0,
                "keypoints": [val[0] for kp in keypoints for val in kp],
                "num_keypoints": len(keypoints),
                "theta": orientations,
            }
        )
        image.save(root / "images" / (str(idx).zfill(zfill) + ".jpg"))
