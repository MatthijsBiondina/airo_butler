import sys
import torchvision.models as models
import torch.nn as nn
import timm
import rospy as ros
from pairo_butler.utils.tools import pyout
from pairo_butler.keypoint_detection.models.backbones.backbone_factory import (
    BackboneFactory,
)


def load_timm_model(model: str):

    backbone = BackboneFactory.create_backbone(model)
    return backbone
