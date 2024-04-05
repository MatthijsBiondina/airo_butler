import itertools
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from PIL import Image
import yaml
from pairo_butler.plotting.heatmap_stream import HeatmapStream
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.keypoint_model.keypoint_dnn import KeypointNeuralNetwork
from pairo_butler.keypoint_model.keypoint_dataset import KeypointDataset
from pairo_butler.utils.tools import (
    UGENT,
    listdir,
    load_config,
    load_mp4_video,
    pbar,
    poem,
    pyout,
)
import rospy as ros


class KeypointModelTrainer:
    def __init__(self, name: str = "keypoint_model_trainer"):
        self.node_name: str = name
        self.config = load_config()

        self.__init_wandb_run()
        self.device = torch.device(self.config.device)
        # self.criterion = nn.functional.mse_loss  # bce_loss
        self.loss_func = nn.functional.binary_cross_entropy_with_logits
        self.batch_size = self.config.batch_size
        self.model, self.optim = self.__initialize_model()
        self.train_loader, self.valid_loader = self.__load_datasets()

        self.indexes, self.index_permutations = self.__init_permutations()

    def start_ros(self):
        ros.init_node(
            f"{self.node_name}_{self.config.backbone}",
            anonymous=True,
            log_level=ros.INFO,
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        step = 0
        best_loss, best_epoch = None, 0

        self.model.train()
        ema = None
        try:
            for epoch in (bar1 := pbar(range(self.config.epochs))):
                self.__check_early_stopping_criteria(epoch, best_epoch)

                for X, t in (
                    bar2 := pbar(self.train_loader, total=len(self.train_loader))
                ):
                    step, ema, y = self.__do_train_step(X, t, ema, step)
                    bar2.desc = poem(f"Train: {ema:.5f}")

                    if step % (len(self.valid_loader.dataset)) < self.batch_size:
                        self.__log_image(X, y, t, "Training")
                        best_loss, best_epoch = self.__do_evaluation_procedure(
                            epoch, best_loss, best_epoch
                        )
                        bar1.desc = poem(f"Best: {best_loss:.5f}")
                        bar1.update(0)

                    wandb.log({"train_loss": ema})

        except BreakException:
            ros.loginfo(f"Exiting training loop.")

    def __init_permutations(self):
        indexes = list(range(self.config.max_nr_of_keypoints))
        permutations_list = list(itertools.permutations(indexes))

        indexes_tensor = torch.stack(
            (torch.tensor(indexes).to(self.device),) * self.config.max_nr_of_keypoints,
            dim=0,
        )
        permutations_tensor = torch.tensor(permutations_list).to(self.device)
        return indexes_tensor, permutations_tensor

    def __init_wandb_run(self):
        wandb.init(project=self.config.project, config=self.config)

    def __initialize_model(self):
        model = KeypointNeuralNetwork(backbone=self.config.backbone).to(self.device)
        optim = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )

        return model, optim

    def __load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        train_set = KeypointDataset(
            root=Path(self.config.root_folder) / "train",
            config=self.config,
            augment=True,
        )
        validation_set = KeypointDataset(
            root=Path(self.config.root_folder) / "validation",
            config=self.config,
            augment=False,
            validation=True,
        )

        if self.batch_size is None:
            self.batch_size = self.__optimize_batch_size(train_set)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        valid_loader = DataLoader(
            validation_set, batch_size=self.batch_size, shuffle=False, num_workers=2
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
            loss = self.criterion_train(y, t)
            loss.backward()
            self.optim.step()
            break

    def __check_early_stopping_criteria(self, epoch, best_epoch):
        if epoch - best_epoch > self.config.patience:
            ros.loginfo(f"Eary stop after {epoch+1} epochs. Ran out of patience.")
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
        loss = self.criterion_train(y, t)
        loss.backward()
        self.optim.step()

        ema = loss if ema is None else alpha * loss.item() + (1 - alpha) * ema
        step += X.shape[0]

        return step, ema, y

    def __log_image(
        self, X: torch.Tensor, y: torch.Tensor, t: torch.Tensor, title: str
    ):
        y, t = y.sum(dim=1, keepdim=True), t.sum(dim=1, keepdim=True)
        y, t = torch.clamp(y, 0, 1), torch.clamp(t, 0, 1)

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

    def __determine_validation_loss(self, subset=False):
        self.model.eval()
        with torch.no_grad():
            cumulative_loss, num = 0.0, 0.0
            for X, t in (bar3 := pbar(self.valid_loader)):
                X, t = X.to(self.device), t.to(self.device)
                y = self.model(X)
                loss = self.criterion_eval(y, t)
                cumulative_loss += loss.cpu().item() * X.shape[0]
                num += X.shape[0]
                bar3.desc = poem(f"Eval: {cumulative_loss / num:.5f}")
                if subset:
                    break
        self.model.train()
        if subset:
            wandb.log({"intermediate_validation_loss": cumulative_loss / num})
        else:
            wandb.log({"validation_loss": cumulative_loss / num})

        self.__log_image(X, y, t, "Validation")

        return cumulative_loss / num

    def __save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "backbone": self.config.backbone,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "loss": loss,
        }
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / (wandb.run.name + ".pth"))

    def criterion_train(self, y: torch.Tensor, t: torch.Tensor):
        return nn.functional.mse_loss(y, t, reduction="mean")

        # # Extract the dimensions of the prediction tensor.
        # bs, n_channels, _, _ = y.shape

        # # Temporarily disable gradient computation to save memory and computations
        # # for this part of the code that's used for determining the best matching
        # # between prediction and target channels.
        # with torch.no_grad():
        #     # Initialize a tensor to hold the computed losses for all possible
        #     # pairings of channels between the prediction and target tensors.
        #     losses = torch.zeros((bs, n_channels, n_channels), device=y.device)

        #     # Calculate the loss for every possible pairing of prediction and target
        #     # channels. This double loop iterates over all pairs of channels.
        #     for ii in range(n_channels):
        #         for jj in range(n_channels):
        #             # Compute the loss between channel ii of the prediction and channel jj of the target
        #             # across all batch items. The loss is calculated pixel-wise and then averaged
        #             # over the spatial dimensions (height and width).
        #             losses[:, ii, jj] = self.loss_func(
        #                 y[:, ii], t[:, jj], reduction="none"
        #             ).mean(dim=(1, 2))

        #     # Calculate the sum of losses for each permutation of channel pairings
        #     # and find the permutation with the minimum loss for each item in the batch.
        #     # `self.indexes` and `self.index_permutations` are used to index into the
        #     # computed losses to consider all possible permutations.
        #     permutation_losses = losses[:, self.indexes, self.index_permutations].sum(
        #         -1
        #     )
        #     permutations_argmin = torch.argmin(permutation_losses, dim=1)

        #     # Based on the selected permutations, find the corresponding channel indexes
        #     # for the target tensor that match best with the prediction tensor's channels.
        #     matching_heatmap_indexes = self.index_permutations[permutations_argmin]

        # # Re-index the target tensor based on the selected channel permutations
        # # to align it with the prediction tensor's channels. This forms the target tensor `t_`
        # # that will be used for the final loss calculation.
        # t_reindexed = t[
        #     torch.arange(bs, device=self.device)[:, None], matching_heatmap_indexes
        # ]

        # # Calculate the final loss between the prediction tensor and the re-indexed
        # # target tensor using the specified loss function. This loss is computed with
        # # gradients enabled, allowing backpropagation for model training.
        # loss = self.loss_func(y, t_reindexed, reduction="mean")

        # # The final calculated loss is returned from the function.
        # return loss

    def criterion_eval(self, y: torch.Tensor, t: torch.Tensor):
        return nn.functional.mse_loss(y, t, reduction="mean")
        y = y.sum(dim=1)
        t = t.sum(dim=1)

        loss = nn.functional.mse_loss(y, t, reduction="mean")
        return loss


import signal
import sys


signal.signal(signal.SIGINT, lambda *args, **kwargs: sys.exit(0))


def main():
    node = KeypointModelTrainer()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
