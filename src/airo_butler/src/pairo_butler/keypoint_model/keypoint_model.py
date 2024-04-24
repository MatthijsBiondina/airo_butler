import time
import torch
from pairo_butler.utils.pods import ArrayPOD, ImagePOD, publish_pod
from pairo_butler.camera.rs2_camera import RS2Client
from pairo_butler.keypoint_model.keypoint_dnn import KeypointNeuralNetwork
import rospy as ros
from airo_butler.msg import PODMessage
from pairo_butler.utils.tools import load_config, pyout
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


class KeypointDNN:
    RATE = 60
    QUEUE_SIZE = 2

    def __init__(self, name: str = "keypoint_dnn"):
        self.node_name: str = name
        self.config = load_config()

        self.device = torch.device(self.config.device)
        self.model = self.__initialize_model()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((512, 512))]
        )

        self.rs2: RS2Client
        self.timestamp: ros.Time
        self.rate: ros.Rate

        self.publisher: ros.Publisher

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.rs2 = RS2Client()
        self.timestamp = ros.Time.now()
        self.rate = ros.Rate(self.RATE)

        self.publisher = ros.Publisher(
            "/keypoints_heatmap", PODMessage, queue_size=self.QUEUE_SIZE
        )

        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        with torch.no_grad():
            while not ros.is_shutdown():
                img_pod: ImagePOD = self.rs2.pod
                if img_pod.timestamp == self.timestamp:
                    self.rate.sleep()
                    continue
                else:
                    try:
                        self.timestamp = img_pod.timestamp
                        heatmap = self.classify(img_pod.image)
                        pod = ArrayPOD(array=heatmap, timestamp=img_pod.timestamp)
                        publish_pod(self.publisher, pod)
                    except Exception as e:
                        ros.logwarn(
                            f"Dropped frame in keypoint classification. (reason: {e})"
                        )

    def classify(self, img: Image.Image):
        X = self.transform(img)
        X = X.to(self.device)[None, ...]
        y = self.model(X).squeeze(1).squeeze(0).cpu().numpy()
        heatmap = np.zeros((y.shape[0], img.height, img.width))
        for ii in range(y.shape[0]):
            heatmap[ii] = cv2.resize(y[ii], (img.width, img.height))
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        return heatmap

    def __initialize_model(self):
        checkpoint_path = f"{self.config.checkpoint_dir}/{self.config.model}"
        checkpoint = torch.load(checkpoint_path)
        model = KeypointNeuralNetwork(backbone=checkpoint["backbone"])
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model


def main():
    node = KeypointDNN()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
