from datetime import datetime
import json
from pathlib import Path
import PIL
import PIL.Image
import numpy as np
from pairo_butler.camera.zed_camera import ZEDClient
from pairo_butler.procedures.subprocedures.drop_towel import DropTowel
from pairo_butler.utils.tools import makedirs, pyout
from pairo_butler.procedures.subprocedure import Subprocedure
import rospy as ros


class RecordTowelSurface(Subprocedure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.zed = ZEDClient()

    def run(self, amount_of_time, nr_of_tries):

        root = Path(self.config.surface_images_folder)
        makedirs(root)

        # Get current time as a ROS Time object
        ros_time = ros.Time.now()

        # Convert ROS Time to seconds since epoch (float)
        secs_since_epoch = ros_time.to_sec()

        # Convert to datetime object
        dt_object = datetime.fromtimestamp(secs_since_epoch)

        # Format datetime object into a string
        folder = root / dt_object.strftime("%Y-%m-%d_%H-%M-%S.%f")
        makedirs(folder)

        rgb: np.ndarray = np.clip(self.zed.pod.rgb_image, 0.0, 1.0) * 255
        img = PIL.Image.fromarray(rgb.astype(np.uint8))

        img.save(folder / "rgb.jpg")

        depth: np.ndarray = self.zed.pod.depth_map
        np.save(folder / "depth_map.npy", depth)

        D = {"time": amount_of_time, "tries": nr_of_tries}
        with open(folder / "stats.json", "w+") as f:
            json.dump(D, f)
