from multiprocessing import Queue
import os
import pickle
from typing import Optional
import warnings

import rospkg
from airo_butler.msg import PODMessage
from torchvision.transforms.functional import to_tensor
from pairo_butler.utils.pods import ArrayPOD, ImagePOD, publish_pod
from pairo_butler.keypoint_detection.utils.load_checkpoints import load_from_checkpoint
import rospy as ros
import torch


class KeypointDetectionDNN:
    QUEUE_SIZE: int = 2
    PUBLISH_RATE: int = 60

    def __init__(
        self,
        name: str = "keypoint_detection",
        checkpoint_name: str = "neat-disco.ckpt",
    ) -> None:
        self.node_name: str = name
        self.rate: Optional[ros.Rate] = None
        self.subscriber: Optional[ros.Subscriber] = None
        self.publisher: Optional[ros.Publisher] = None

        self.pod: Optional[ImagePOD] = None
        self.model = self.__init_model(checkpoint_name)

    def __init_model(self, checkpoint: str):
        rospack = rospkg.RosPack()
        checkpoint_path = os.path.join(
            rospack.get_path("airo_butler"),
            "res/models/keypoint_detection",
            checkpoint,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_from_checkpoint(checkpoint_path)
        model.eval()
        model.cuda()
        return model

    def start_ros(self) -> None:
        ros.init_node(self.node_name, log_level=ros.INFO)
        self.rate = ros.Rate(self.PUBLISH_RATE)
        self.subscriber = ros.Subscriber(
            "/color_frame", PODMessage, self.__sub_callback, queue_size=self.QUEUE_SIZE
        )
        self.publisher = ros.Publisher(
            "/heatmap", PODMessage, queue_size=self.QUEUE_SIZE
        )
        ros.loginfo(f"{self.node_name}: OK!")

    def __sub_callback(self, msg: PODMessage):
        """Callback function for receiving POD

        Args:
            msg (PODMessage): plain old data containing image and timestamp
        """
        self.pod = pickle.loads(msg.data)

    def run(self):
        while not ros.is_shutdown():
            if self.pod is None:
                pass
            else:
                pod_in: ImagePOD = self.pod
                self.pod = None
                image = to_tensor(pod_in.image)[None, ...].cuda()
                with torch.no_grad():
                    heatmap = self.model(image).squeeze()
                heatmap = heatmap.cpu().numpy()

                pod_ou = ArrayPOD(array=heatmap, timestamp=pod_in.timestamp)
                publish_pod(self.publisher, pod_ou)
            self.rate.sleep()


def main():
    node = KeypointDetectionDNN()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
