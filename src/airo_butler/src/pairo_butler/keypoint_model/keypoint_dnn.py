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

        self.line_1 = nn.Conv2d(2048, out_channels=1024, kernel_size=1)
        self.line_2 = nn.Conv2d(1024, out_channels=1, kernel_size=1)

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

        hidden_layer = torch.relu(self.line_1(concatenated_feature_map))
        output = torch.sigmoid(self.line_2(hidden_layer))

        return output
