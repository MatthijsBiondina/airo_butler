from pairo_butler.utils.tools import load_config
from pairo_butler.keypoint_model.pretrained_models import (
    load_timm_model,
)
import torch.nn as nn
import torch.nn.functional as F
import torch


class KeypointNeuralNetwork(nn.Module):
    def __init__(self, backbone: str):
        super(KeypointNeuralNetwork, self).__init__()
        config = load_config()

        self.backbone = load_timm_model(backbone)
        self.head = nn.Conv2d(
            in_channels=self.backbone.get_n_channels_out(),
            out_channels=1,
            kernel_size=(3, 3),
            padding="same",
        )
        self.head.bias.data.fill_(-4)

    def forward(self, x):

        h = self.backbone(x)
        h = self.head(h)

        return torch.sigmoid(h)
