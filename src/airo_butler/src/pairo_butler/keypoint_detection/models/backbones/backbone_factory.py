import argparse
from typing import List

from pairo_butler.keypoint_detection.models.backbones.base_backbone import Backbone
from pairo_butler.keypoint_detection.models.backbones.convnext_unet import ConvNeXtUnet
from pairo_butler.keypoint_detection.models.backbones.dilated_cnn import DilatedCnn
from pairo_butler.keypoint_detection.models.backbones.maxvit_unet import (
    MaxVitPicoUnet,
    MaxVitUnet,
)
from pairo_butler.keypoint_detection.models.backbones.mobilenetv3 import MobileNetV3
from pairo_butler.keypoint_detection.models.backbones.s3k import S3K
from pairo_butler.keypoint_detection.models.backbones.unet import Unet


class BackboneFactory:
    # TODO: how to auto-register with __init__subclass over multiple files?
    registered_backbone_classes: List[Backbone] = [
        Unet,
        ConvNeXtUnet,
        MaxVitUnet,
        MaxVitPicoUnet,
        S3K,
        DilatedCnn,
        MobileNetV3,
    ]

    @staticmethod
    def create_backbone(backbone_type: str, **kwargs) -> Backbone:
        for backbone_class in BackboneFactory.registered_backbone_classes:
            if backbone_type == backbone_class.__name__:
                return backbone_class(**kwargs)
        raise Exception(f"Unknown backbone type: {backbone_type}")

    @staticmethod
    def add_to_argparse(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(BackboneFactory.__name__)
        parser.add_argument(
            "--backbone_type",
            type=str,
            default=Unet.__name__,
            help="The Class of the Backbone for the Detector.",
        )
        # add all backbone hyperparams.
        for backbone_class in BackboneFactory.registered_backbone_classes:
            parent_parser = backbone_class.add_to_argparse(parent_parser)
        return parent_parser
