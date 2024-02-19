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

        self.backbone, feature_channels = load_timm_model(backbone)

        self.reduce_channels_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1)
                for channels in feature_channels
            ]
        )

        self.line_1 = nn.Conv2d(
            128 * len(feature_channels), out_channels=128, kernel_size=1
        )
        self.line_2 = nn.Conv2d(128, out_channels=1, kernel_size=1)

    def forward(self, x):

        feature_maps = self.backbone(x)

        # Reduce channel dimensions and upscale each feature map
        reduced_and_upscaled_maps = []
        for layer, feature_map in zip(self.reduce_channels_layers, feature_maps):
            reduced_map = layer(feature_map)  # Reduce channels to 128
            upscaled_map = F.interpolate(
                reduced_map, size=(512, 512), mode="bilinear", align_corners=False
            )
            reduced_and_upscaled_maps.append(upscaled_map)

        # Concatenate the reduced and upscaled feature maps
        concatenated_feature_map = torch.cat(reduced_and_upscaled_maps, dim=1)

        hidden_layer = torch.relu(self.line_1(concatenated_feature_map))
        output = torch.sigmoid(self.line_2(hidden_layer))

        return output
