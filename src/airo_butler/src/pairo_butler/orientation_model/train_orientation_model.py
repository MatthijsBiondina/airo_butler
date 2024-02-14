import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import PIL
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import rospkg
import wandb
import yaml
from pairo_butler.orientation_model.orientation_utils import OrientationUtils
from pairo_butler.orientation_model.orientation_resnet import OrientationNeuralNetwork
from pairo_butler.orientation_model.orientation_dataset import OrientationDataset
from pairo_butler.labelling.coco_dataset_structure import COCODatasetStructure
from pairo_butler.labelling.labelling_utils import LabellingUtils
from pairo_butler.labelling.determine_visibility import VisibilityChecker
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, poem, pyout
import rospy as ros


class OrientationModelTrainer:
    def __init__(self, name: str = "orientation_model_trainer"):
        self.node_name: str = name

        config_path: Path = Path(__file__).parent / "orientation_config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.__init_wandb_run()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.valid_loader = self.__load_datasets()
        self.model, self.optim = self.__initialize_model()

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        self.__training_loop()

    def __init_wandb_run(self):
        wandb.init(project=self.config["project"], config=self.config)

    def __load_datasets(self):
        train_set = OrientationDataset(
            root=Path(self.config["root_folder"]) / "train",
            size=self.config["size"],
            heatmap_sigma=self.config["heatmap_sigma"],
            heatmap_size=self.config["heatmap_size"],
            augment=True,
        )
        valid_set = OrientationDataset(
            root=Path(self.config["root_folder"]) / "validation",
            size=self.config["size"],
            heatmap_sigma=self.config["heatmap_sigma"],
            heatmap_size=self.config["heatmap_size"],
            augment=False,
        )

        train_loader = DataLoader(
            train_set, batch_size=self.config["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=self.config["batch_size"], shuffle=False
        )

        return train_loader, valid_loader

    def __initialize_model(self):
        model = OrientationNeuralNetwork(
            num_classes=self.config["heatmap_size"],
            dropout_rate=self.config["dropout_rate"],
        ).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        return model, optim

    def __training_loop(self):
        step = 0
        best_loss, best_epoch = None, 0

        for epoch in (bar1 := pbar(range(self.config["epochs"]))):
            if epoch - best_epoch > self.config["patience"]:
                break

            self.model.train()
            ema_loss = None
            for ii, (X, t) in (
                bar2 := pbar(enumerate(self.train_loader), total=len(self.train_loader))
            ):
                self.optim.zero_grad()
                X, t = X.to(self.device), t.to(self.device)
                y = self.model(X)
                loss = torch.nn.functional.mse_loss(y, t)
                loss.backward()
                self.optim.step()
                ema_loss = loss if ema_loss is None else 0.1 * loss + 0.9 * ema_loss
                bar2.desc = poem(f"Train:   {ema_loss:.5f}")

                step += X.shape[0]
                wandb.log({"train_step_loss": loss.item(), "step": step})

                if step % 1000 < self.config["batch_size"]:
                    OrientationUtils.plot_to_wandb(X, y, t, title="train_plot")

                    self.model.eval()
                    with torch.no_grad():
                        cum_loss, num = 0.0, 0.0
                        for X, t in (bar3 := pbar(self.valid_loader)):
                            X, t = X.to(self.device), t.to(self.device)
                            y = self.model(X)
                            loss = torch.nn.functional.mse_loss(y, t)
                            cum_loss += loss.cpu().item() * X.shape[0]
                            num += X.shape[0]
                            bar3.desc = poem(f"Eval: {cum_loss / num:.5f}")
                    wandb.log({"eval_loss": cum_loss / num})
                    OrientationUtils.plot_to_wandb(X, y, t, title="eval_plot")

                    new_loss = cum_loss / num
                    if best_loss is None or new_loss < best_loss:
                        best_loss = new_loss
                        best_epoch = epoch
                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optim.state_dict(),
                            "loss": new_loss,
                        }
                        checkpoint_dir = Path(f"/home/matt/Models/orientation")
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            checkpoint, checkpoint_dir / (wandb.run.name + ".pth")
                        )
                        bar1.desc = poem(f"Best: {best_loss:.5f}")
                        bar1.update(0)
                    self.model.train()
            wandb.log({"train_loss": ema_loss.item()})


def main():
    node = OrientationModelTrainer()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
