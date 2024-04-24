import abc
import argparse
from pathlib import Path

from munch import Munch, munchify
import rospkg
from torch import nn as nn
import yaml


class Backbone(nn.Module, abc.ABC):
    """Base class for backbones"""

    def __init__(self):
        super(Backbone, self).__init__()
        config_path = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "src"
            / "pairo_butler"
            / "keypoint_model"
            / "config.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = Munch.fromDict(config)

    @abc.abstractmethod
    def get_n_channels_out(self) -> int:
        raise NotImplementedError

    @staticmethod
    def add_to_argparse(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        return parent_parser
