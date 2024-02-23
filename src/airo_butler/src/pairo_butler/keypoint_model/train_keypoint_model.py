from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from PIL import Image
import yaml
import matplotlib.pyplot as plt
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.keypoint_model.keypoint_dnn import KeypointNeuralNetwork
from pairo_butler.keypoint_model.keypoint_dataset import KeypointDataset
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, poem, pyout

# import rospy as ros


class HeatmapStream:
    @staticmethod
    def overlay_heatmap_on_image(image: Image, heatmap: np.ndarray):

        # Normalize heatmap to [0, 1] range based on its min and max values
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        heatmap_normalized = (heatmap - heatmap_min) / (
            heatmap_max - heatmap_min + 1e-6
        )

        colormap = plt.get_cmap("viridis")
        heatmap_colored = colormap(heatmap_normalized)

        # Convert to PIL image and ensure same size as original
        heatmap_image = Image.fromarray(
            (heatmap_colored * 255).astype(np.uint8)
        ).convert("RGBA")
        heatmap_image.putalpha(100)

        # Overlay the heatmap on the image
        overlay_image = Image.new("RGBA", image.size)
        overlay_image = Image.alpha_composite(overlay_image, image.convert("RGBA"))
        overlay_image = Image.alpha_composite(overlay_image, heatmap_image)

        # Convert to rgb
        background = Image.new("RGB", overlay_image.size, (255, 255, 255))
        rgb_image = Image.alpha_composite(
            background.convert("RGBA"), overlay_image
        ).convert("RGB")

        return rgb_image


class KeypointModelTrainer:
    def __init__(self, name: str = "keypoint_model_trainer"):
        self.node_name: str = name
        config_path: Path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.__init_wandb_run()
        self.device = torch.device("cuda:0")
        # self.device = torch.device("cpu")
        self.criterion = nn.functional.mse_loss
        self.batch_size = self.config["batch_size"]
        self.model, self.optim = self.__initialize_model()
        self.train_loader, self.valid_loader = self.__load_datasets()

    def start_ros(self):
        pass
        # ros.init_node(self.node_name, log_level=ros.INFO)
        # ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        step = 0
        best_loss, best_epoch = None, 0

        self.model.train()
        ema = None
        try:
            for epoch in (bar1 := pbar(range(self.config["epochs"]))):
                self.__check_early_stopping_criteria(epoch, best_epoch)

                for X, t in (
                    bar2 := pbar(self.train_loader, total=len(self.train_loader))
                ):
                    step, ema, y = self.__do_train_step(X, t, ema, step)
                    bar2.desc = poem(f"Train: {ema:.5f}")

                    if step % len(self.valid_loader.dataset) < self.batch_size:
                        self.__log_image(X, y, t, "Training")
                        best_loss, best_epoch = self.__do_evaluation_procedure(
                            epoch, best_loss, best_epoch
                        )
                        bar1.desc = poem(f"Best: {best_loss:.5f}")
                        bar1.update(0)

                wandb.log({"train_loss": ema})

        except BreakException:
            pyout(f"Exiting training loop.")
            # ros.loginfo(f"Exiting training loop.")

    def __init_wandb_run(self):
        wandb.init(project=self.config["project"], config=self.config)

    def __initialize_model(self):
        model = KeypointNeuralNetwork(backbone=self.config["backbone"]).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        return model, optim

    def __load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        train_set = KeypointDataset(
            root=Path(self.config["root_folder"]) / "train",
            config=self.config,
            augment=True,
        )
        validation_set = KeypointDataset(
            root=Path(self.config["root_folder"]) / "validation",
            config=self.config,
            augment=False,
        )

        if self.batch_size is None:
            self.batch_size = self.__optimize_batch_size(train_set)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        valid_loader = DataLoader(
            validation_set, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

        return train_loader, valid_loader

    def __optimize_batch_size(self, dataset: KeypointDataset):
        batch_size = 1
        successful_batch_size = batch_size
        max_batch_size_found = False

        while not max_batch_size_found:
            try:
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.__test_batch_size(data_loader)
                successful_batch_size = batch_size
                batch_size *= 2
            except RuntimeError as e:
                if "out of memory" in str(e):
                    pyout(
                        f"OOM at batch size: {batch_size}, rolling back to {successful_batch_size}"
                    )
                    max_batch_size_found = True
                else:
                    raise e  # Re-raise exception if it's not an OOM error

        return successful_batch_size

    def __test_batch_size(self, data_loader: DataLoader) -> None:
        for X, t in data_loader:
            X, t = X.to(self.device), t.to(self.device)
            self.optim.zero_grad()
            y = self.model(X)
            loss = self.criterion(y, t)
            loss.backward()
            # self.optim.step()
            break

    def __check_early_stopping_criteria(self, epoch, best_epoch):
        if epoch - best_epoch > self.config["patience"]:
            pyout(f"Eary stop after {epoch+1} epochs. Ran out of patience.")
            # ros.loginfo(f"Eary stop after {epoch+1} epochs. Ran out of patience.")
            raise BreakException

    def __do_train_step(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        ema: Optional[float],
        step: int,
        alpha: float = 0.01,
    ) -> Tuple[int, float]:
        self.optim.zero_grad()
        X, t = X.to(self.device), t.to(self.device)
        y = self.model(X)
        loss = self.criterion(y, t)
        loss.backward()

        # Clip gradients to avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)

        # Iterate through model parameters and set NaN gradients to 0
        for param in self.model.parameters():
            if param.grad is not None:
                with torch.no_grad():
                    param.grad[param.grad != param.grad] = (
                        0.0  # Sets NaN gradients to 0
                    )

        self.optim.step()

        ema = loss if ema is None else alpha * loss.item() + (1 - alpha) * ema
        step += X.shape[0]

        return step, ema, y

    def __log_image(
        self, X: torch.Tensor, y: torch.Tensor, t: torch.Tensor, title: str
    ):
        idx = max(list(range(t.shape[0])), key=lambda i: torch.max(t[i]).item())

        image = Image.fromarray(
            (X[idx].detach().cpu().numpy().clip(0, 1).transpose(1, 2, 0) * 255).astype(
                np.uint8
            )
        )
        heatmap_target: np.ndarray = t[idx, 0].detach().cpu().numpy()
        heatmap_predicted: np.ndarray = y[idx, 0].detach().cpu().numpy()

        image_target = HeatmapStream.overlay_heatmap_on_image(image, heatmap_target)
        image_predicted = HeatmapStream.overlay_heatmap_on_image(
            image, heatmap_predicted
        )

        # Stick images side by side
        # Calculate dimensions for the new image
        total_width = image_target.width + image_predicted.width
        max_height = max(image_target.height, image_predicted.height)

        # Create a new image with appropriate dimensions
        joined_images = Image.new("RGB", (total_width, max_height))

        # Paste the two images into the new image
        joined_images.paste(image_target, (0, 0))
        joined_images.paste(image_predicted, (image_target.width, 0))

        wandb.log({title: wandb.Image(joined_images)})

    def __do_evaluation_procedure(
        self, current_epoch: int, best_loss: Optional[float], best_epoch: int
    ):
        new_loss = self.__determine_validation_loss()

        if best_loss is None or new_loss < best_loss:
            self.__save_checkpoint(current_epoch, new_loss)
            best_loss = new_loss
            best_epoch = current_epoch

        return best_loss, best_epoch

    def __determine_validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            cumulative_loss, num = 0.0, 0.0
            for X, t in (bar3 := pbar(self.valid_loader)):
                X, t = X.to(self.device), t.to(self.device)
                y = self.model(X)
                loss = self.criterion(y, t)
                cumulative_loss += loss.cpu().item() * X.shape[0]
                num += X.shape[0]
                bar3.desc = poem(f"Eval: {cumulative_loss / num:.5f}")
        self.model.train()
        wandb.log({"validation_loss": cumulative_loss / num})

        self.__log_image(X, y, t, "Validation")

        return cumulative_loss / num

    def __save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "loss": loss,
        }
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / (wandb.run.name + ".pth"))


def main():
    node = KeypointModelTrainer()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
