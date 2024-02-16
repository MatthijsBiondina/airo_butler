import json
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from pairo_butler.orientation_model.orientation_pods import Coord
from pairo_butler.keypoint_model.keypoint_model_pods import KeypointSampleMetaDataPOD

from pairo_butler.utils.tools import pbar, pyout


class KeypointDataset(Dataset):
    def __init__(
        self,
        root: Path,
        config: Dict[str, Any],
        augment: bool = False,
    ):
        self.root = root
        self.config = config
        self.augment = augment

        self.image_transform = self.__init_transform(augment=augment)
        self.heatmap_transform = self.__init_transform()
        self.samples: List[KeypointSampleMetaDataPOD] = self.__init_data()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[idx]

        image = Image.open(sample.path)
        keypoints = sample.keypoints
        heatmap = self.__make_heatmap(keypoints, (image.width, image.height))

        image, heatmap = self.__preprocess_image_and_heatmap(image, heatmap)

        return image, heatmap

    def __init_data(self) -> List[KeypointSampleMetaDataPOD]:
        samples = []

        with open(self.root / "annotations.json", "r") as f:
            annotations = json.load(f)

        for image, annotation in pbar(
            zip(annotations["images"], annotations["annotations"]),
            desc="Loading Data",
            total=len(annotations["annotations"]),
        ):
            assert image["id"] == annotation["id"]

            img_path = self.root / image["file_name"]

            keypoints = []
            for kp_idx in range(0, len(annotation["keypoints"]), 3):
                x = annotation["keypoints"][kp_idx]
                y = annotation["keypoints"][kp_idx + 1]
                visibility = annotation["keypoints"][kp_idx + 2]

                if visibility == 2.0:
                    keypoints.append(Coord(x=x, y=y))
            samples.append(
                KeypointSampleMetaDataPOD(path=img_path, keypoints=keypoints)
            )
        return samples

    def __make_heatmap(
        self, keypoints: List[Coord], img_size: Tuple[int, int]
    ) -> Image.Image:
        w, h = img_size
        heatmap = np.zeros((h, w), dtype=np.float32)

        for point in keypoints:
            # Generate a grid of (x,y) coordinates
            x = np.arange(0, w, 1, np.float32)
            y = np.arange(0, h, 1, np.float32)
            y = y[:, np.newaxis]

            # Calculate the 2D Gaussian
            heatmap += np.exp(
                -((x - point.x) ** 2 + (y - point.y) ** 2)
                / (2 * self.config["heatmap_sigma"] ** 2)
            )

        # Normalize the heatmap to [0, 1]
        heatmap = np.clip(heatmap, 0, 1)

        # Convert heatmap to an image
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))

        return heatmap_img

    def __init_transform(self, augment: bool = False) -> Compose:
        if augment:
            return transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            return transforms.Compose([transforms.ToTensor()])

    def __preprocess_image_and_heatmap(self, image: Image.Image, heatmap: Image.Image):
        if self.augment:
            image, heatmap = self.__random_rotate(image, heatmap)
            image, heatmap = self.__random_crop(image, heatmap)

        X = self.image_transform(image)
        t = self.heatmap_transform(heatmap)

        return X, t

    def __random_crop(
        self, image: Image.Image, heatmap: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        crop_width = np.random.randint(
            int(self.config["min_crop_ratio"] * image.width), image.width
        )
        crop_height = np.random.randint(
            int(self.config["min_crop_ratio"] * image.height), image.height
        )

        left = np.random.randint(0, image.width - crop_width)
        top = np.random.randint(0, image.height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        image_cropped = image.crop((left, top, right, bottom))
        heatmap_cropped = heatmap.crop((left, top, right, bottom))

        image_resized = image_cropped.resize((image.width, image.height))
        heatmap_resized = heatmap_cropped.resize((image.width, image.height))

        return image_resized, heatmap_resized

    def __random_rotate(
        self, image: Image.Image, heatmap: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        angle_degrees = np.random.uniform(
            -self.config["max_rotate_angle"], self.config["max_rotate_angle"]
        )

        image_rotated = image.rotate(angle_degrees, expand=False, fillcolor=(0, 0, 0))
        heatmap_rotated = heatmap.rotate(angle_degrees, expand=False, fillcolor=0)
        return image_rotated, heatmap_rotated
