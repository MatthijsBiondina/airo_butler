import sys
import torchvision.models as models
import torch.nn as nn
import timm
import rospy as ros
from pairo_butler.utils.tools import pyout


def load_timm_model(model: str):
    pyout(f"Loading pretrained model: {model}")
    backbone = timm.create_model(model, pretrained=True, features_only=True)
    for idx, feature in enumerate(backbone.feature_info):
        num_channels = feature["num_chs"]
        pyout(f"Output {idx}: {num_channels} channels")
    sys.exit(0)

    return backbone


def load_pretrained_model(model: str):

    if model == "maxvit_large_tf_512":
        backbone = timm.create_model("maxvit_large_tf_512", pretrained=True)

        pyout()

    if model == "resnet50":
        backbone = models.resnet50(pretrained=True)
        # Remove the fully connected layer (and possibly the avgpool layer depending on your needs)
        backbone_features = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # assuming we keep the avgpool layer

        # Define the decoder
        decoder = nn.Sequential(
            # The first ConvTranspose2d matches ResNet-50's output feature size (2048 channels)
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=1),
        )
        # Combine backbone and decoder
        combined = nn.Sequential(
            *list(backbone_features.children()) + list(decoder.children())
        )
        return combined

    if model == "resnet101":
        backbone = models.resnet101(pretrained=True)
        # Remove the fully connected layer (and possibly the avgpool layer depending on your needs)
        backbone_features = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # assuming we keep the avgpool layer

        # Define the decoder
        decoder = nn.Sequential(
            # Adjust the first ConvTranspose2d to match ResNet-101's output feature size
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=1),
        )
        # Combine backbone and decoder
        combined = nn.Sequential(
            *list(backbone_features.children()) + list(decoder.children())
        )
        return combined

    if model == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(pretrained=True)
        backbone_features = nn.Sequential(*(list(backbone.children())[:-2]))
        decoder = nn.Sequential(
            # Assuming the encoder features size is 1/32 of the input
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=1),
        )
        combined = nn.Sequential(
            *list(backbone_features.children()) + list(decoder.children())
        )

        return combined

    if model == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(pretrained=True)
        backbone_features = nn.Sequential(*(list(backbone.children())[:-2]))
        decoder = nn.Sequential(
            # Assuming the encoder features size is 1/32 of the input
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=1),
        )
        combined = nn.Sequential(
            *list(backbone_features.children()) + list(decoder.children())
        )

        return combined

    raise ValueError(f"Unknown backbone: {model}")
