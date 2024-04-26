from typing import Tuple
import torch
from pairo_butler.data.timesync import TimeSync
from pairo_butler.orientation_model.orientation_resnet import OrientationNeuralNetwork
from pairo_butler.utils.pods import (
    KeypointThetaPOD,
    KeypointUVPOD,
    publish_pod,
)
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.tools import load_config, pyout
from torchvision import transforms
from PIL import Image
import numpy as np


class OrientationDNN:
    """
    A ROS node for determining the orientation of keypoints using a deep neural network (DNN).

    This class integrates a neural network model to process image data and extract
    orientation information about keypoints within those images. The orientations
    are calculated within a circular domain of [-pi, pi].

    Attributes:
        node_name (str): The name of the ROS node.
        config: Configuration settings loaded from an external source.
        rate (ros.Rate): The rate at which the node publishes messages.
        sync (TimeSync): An instance for synchronizing messages from different topics.
        device (torch.device): The device on which the neural network operates.
        model (torch.nn.Module): The neural network model for orientation estimation.
        transforms (torchvision.transforms.Compose): Pre-processing transforms applied to input images.
        publisher (ros.Publisher): A ROS publisher for publishing orientation data.
    """

    RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "orientation_dnn"):

        self.node_name: str = name
        self.config = load_config()
        self.rate: ros.Rate
        self.sync: TimeSync

        self.device = torch.device(self.config.device)
        self.model = self.__initialize_model()
        self.transforms = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )

        self.publisher: ros.Publisher

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.RATE)
        self.sync = TimeSync(
            ankor_topic="/keypoints_uv", unsynced_topics=["/rs2_topic"]
        )

        self.publisher = ros.Publisher(
            "/keypoints_theta", PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        with torch.no_grad():
            while not ros.is_shutdown():
                packages, timestamp = self.sync.next()

                if packages["/keypoints_uv"]["pod"].valid:
                    X = self.__preprocess_img(
                        packages["/rs2_topic"]["pod"].image,
                        packages["/keypoints_uv"]["pod"],
                    ).to(self.device)
                    y = self.model(X[None, ...])
                    mean, stdev = self.__postprocess(y)
                    pod = KeypointThetaPOD(
                        timestamp=timestamp,
                        valid=True,
                        mean=mean,
                        stdev=stdev,
                    )
                else:
                    pod = KeypointThetaPOD(timestamp=timestamp, valid=False)
                publish_pod(self.publisher, pod)

                self.rate.sleep()

    def __initialize_model(self):
        checkpoint_path = f"{self.config.checkpoint_dir}/{self.config.model}"
        checkpoint = torch.load(checkpoint_path)
        model = OrientationNeuralNetwork(
            num_classes=self.config.heatmap_size,
        ).to(self.device)
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def __preprocess_img(self, img: Image.Image, keypoint: KeypointUVPOD):
        crop_box = (
            max(0, keypoint.x - self.config.size // 2),
            max(0, keypoint.y - self.config.size // 2),
            min(img.width, keypoint.x + (self.config.size + 1) // 2),
            min(img.height, keypoint.y + (self.config.size + 1) // 2),
        )

        cropped_img = img.crop(crop_box)

        black_background = Image.new(
            "RGB", (self.config.size, self.config.size), (0, 0, 0)
        )
        black_background.paste(cropped_img, (0, 0))

        return self.transforms(black_background)

    def __postprocess(self, distribution: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of a wrapped Gaussian distribution
        across a circular domain of [-pi, pi].

        This function computes the mean and standard deviation for a probability distribution
        that is assumed to be periodic with a period of 2*pi. The distribution is provided
        as a tensor that represents a histogram of probabilities across angular values
        from -pi to pi.

        Args:
            distribution (torch.Tensor): A 1D tensor representing the probability mass
                                        function over the interval [-pi, pi].

        Returns:
            Tuple[float, float]: The mean and standard deviation of the distribution.
                                The mean is the central angle where the standard deviation,
                                computed over circular distances, is minimized.

        Notes:
            - The distribution must sum to 1 (i.e., represent a valid probability distribution).
            - The function computes a circular (or "wrapped") standard deviation, which
            accounts for the periodicity of the angle domain.
        """
        # Ensure the distribution is a proper probability distribution
        d = distribution.squeeze(0).cpu().numpy()
        d = d / np.sum(d)

        # Create an array of means evenly spaced over [-pi, pi)
        means = np.linspace(-np.pi, np.pi, num=self.config.heatmap_size, endpoint=False)

        # Calculate pairwise circular distances between means
        d_mu = np.abs(means[:, None] - means[None, :])
        d_mu_wrapped = np.minimum(d_mu, 2 * np.pi - d_mu)

        # Calculate the weighted circular variance for each mean
        stdev = (d[None, :] @ d_mu_wrapped).squeeze(0)

        # Find the index of the minimum standard deviation to determine the best mean
        ii_min = np.argmin(stdev)

        # Select the mean and standard deviation at the minimal index
        mean = means[ii_min]
        stdev = stdev[ii_min]

        return (mean, stdev)


def main():
    node = OrientationDNN()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
import cv2