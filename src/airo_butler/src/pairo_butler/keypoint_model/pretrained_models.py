import torchvision.models as models
import torch.nn as nn


def load_pretrained_model(model: str):

    if model == "mobilenet_v3_large":
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
