import sys
from pairo_butler.utils.tools import pyout
from pairo_butler.keypoint_model.pretrained_models import (
    load_pretrained_model,
    load_timm_model,
)
import torch.nn as nn
import torch


class KeypointNeuralNetwork(nn.Module):
    def __init__(self, backbone: str):
        super(KeypointNeuralNetwork, self).__init__()

        self.backbone = load_timm_model(backbone)

    def forward(self, x):

        h = self.backbone(x)

        for h_ in h:
            pyout(h_.shape)

        sys.exit(0)

        return torch.sigmoid(self.backbone(x))
