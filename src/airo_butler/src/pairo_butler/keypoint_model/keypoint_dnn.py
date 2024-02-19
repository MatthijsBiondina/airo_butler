import sys
from pairo_butler.utils.tools import pyout
from pairo_butler.keypoint_model.pretrained_models import (
    load_pretrained_model,
    load_timm_model,
)
import torch.nn as nn
import torch.nn.functional as F
import torch


class KeypointNeuralNetwork(nn.Module):
    def __init__(self, backbone: str):
        super(KeypointNeuralNetwork, self).__init__()

        self.backbone = load_timm_model(backbone)

    def forward(self, x):

        feature_maps = self.backbone(x)

        upscaled_maps = []

        # Upscale each feature map to 512x512
        for feature_map in feature_maps:
            upscaled_map = F.interpolate(
                feature_map, size=(512, 512), mode="bilinear", align_corners=False
            )
            upscaled_maps.append(upscaled_map)

        concatenated_feature_map = torch.cat(upscaled_maps, dim=1)

        pyout(concatenated_feature_map.shape)

        sys.exit(0)

        return torch.sigmoid(self.backbone(x))
