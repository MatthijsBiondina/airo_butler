from pairo_butler.utils.tools import pyout
from pairo_butler.keypoint_model.pretrained_models import load_pretrained_model
import torch.nn as nn
import torch


class KeypointNeuralNetwork(nn.Module):
    def __init__(self, backbone: str):
        super(KeypointNeuralNetwork, self).__init__()

        self.backbone = load_pretrained_model(backbone)
        # self.conv_final = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
        # features = self.backbone(x)
        # output = self.conv_final(features)
        # return output
