import timm
import torch
import torch.nn as nn
from pairo_butler.keypoint_detection.models.backbones.base_backbone import Backbone
from pairo_butler.keypoint_detection.models.backbones.convnext_unet import (
    UpSamplingBlock,
)


class MobileNetV3(Backbone):
    """
    Pretrained MobileNetV3 using the large_100 model with 3.4M parameters.
    Incorporates dropout for regularization in the decoder.
    """

    def __init__(self, **kwargs):
        super().__init__()
        dropout_rate = self.config.dropout

        self.encoder = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, features_only=True
        )
        self.decoder_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # List to hold dropout layers

        for i in range(1, len(self.encoder.feature_info.info)):
            channels_in, skip_channels_in = (
                self.encoder.feature_info.info[-i]["num_chs"],
                self.encoder.feature_info.info[-i - 1]["num_chs"],
            )
            block = UpSamplingBlock(channels_in, skip_channels_in, skip_channels_in, 3)
            self.decoder_blocks.append(block)
            self.dropouts.append(nn.Dropout2d(dropout_rate))  # Add dropout layer

        self.final_conv = nn.Conv2d(
            skip_channels_in, skip_channels_in, 3, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        x = features.pop()
        for block, dropout in zip(self.decoder_blocks, self.dropouts):  # Use dropout
            x = block(x, features.pop())
            x = dropout(x)  # Apply dropout
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.final_conv(x)

        return x

    def get_n_channels_out(self):
        return self.encoder.feature_info.info[0]["num_chs"]
