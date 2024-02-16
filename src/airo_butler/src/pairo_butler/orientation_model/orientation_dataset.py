from pathlib import Path
from pairo_butler.orientation_model.orientation_pods import Coord, SampleMetaDataPOD
from pairo_butler.utils.tools import listdir, pbar, pyout
import json
from random import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose


class OrientationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        size: int,
        heatmap_sigma: float,
        heatmap_size: int,
        augment: bool = False,
    ):
        self.root: Path = root
        self.width: int = size
        self.height: int = size
        self.heatmap_sigma: float = heatmap_sigma
        self.heatmap_size: float = heatmap_size

        self.transform: Compose = self.__init_transform(augment)

        self.samples: List[SampleMetaDataPOD] = self.__init_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        x = self.__open_and_preprocess_image(meta.path, meta.center)
        y = OrientationDataset.wrapped_gaussian_heatmap(
            meta.angle, self.heatmap_sigma, self.heatmap_size
        )

        return x, y

    def __init_transform(self, data_augmentation: bool) -> Compose:
        if data_augmentation:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(256),
                    transforms.RandomRotation(3),
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            return transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    def __init_data(self):
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

            for ii, kp_idx in enumerate(range(0, len(annotation["keypoints"]), 3)):
                x = int(round(annotation["keypoints"][kp_idx]))
                y = int(round(annotation["keypoints"][kp_idx + 1]))
                visibility = int(round(annotation["keypoints"][kp_idx + 2]))

                if visibility == 2:
                    samples.append(
                        SampleMetaDataPOD(
                            path=img_path,
                            center=Coord(x, y),
                            angle=annotation["theta"][ii],
                        )
                    )

        return samples

    def __open_and_preprocess_image(self, path: Path, center: Coord) -> Tensor:
        img = Image.open(path)

        crop_box = (
            max(0, center.x - self.width // 2),  # left
            max(0, center.y - self.height // 2),  # top
            min(img.width, center.x + (self.width + 1) // 2),  # right
            min(img.height, center.y + (self.height + 1) // 2),
        )  # bottom

        cropped_img = img.crop(crop_box)

        black_background = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        black_background.paste(cropped_img, (0, 0))

        return self.transform(black_background)

    @staticmethod
    def wrapped_gaussian_heatmap(mean: float, std_dev: float, N: int) -> Tensor:
        std_dev = np.deg2rad(std_dev)

        x = np.linspace(-np.pi, np.pi, num=N, endpoint=False)

        wrapped_distance = np.minimum(np.abs(x - mean), 2 * np.pi - np.abs(x - mean))

        logits = np.exp(-0.5 * (wrapped_distance / std_dev) ** 2) / (
            std_dev * np.sqrt(2 * np.pi)
        )
        logits = logits / np.max(logits)

        return FloatTensor(logits)
