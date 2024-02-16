from pairo_butler.utils.tools import pyout
from pairo_butler.keypoint_model.pretrained_models import load_pretrained_model
import torch.nn as nn
import torch


class KeypointNeuralNetwork(nn.Module):
    def __init__(self, backbone: str):
        super(KeypointNeuralNetwork, self).__init__()

        self.backbone = load_pretrained_model(backbone)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
