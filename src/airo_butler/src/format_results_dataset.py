import json
import os
from pathlib import Path
import shutil
import cv2

import tifffile
from pairo_butler.ur5e_arms.ur5e_client import UR5eClient
from pairo_butler.camera.zed_camera import ZED
from pairo_butler.utils.tools import listdir, makedirs, pbar, pyout
from PIL import Image
import numpy as np
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=2, suppress=True)

ROOT = Path("/media/matt/Expansion/surface_dataset")
OUT = Path("/media/matt/Expansion/observation_results")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def tcp2json(tcp):
    position = tcp[:3, 3]
    euler_angles = R.from_matrix(tcp[:3, :3]).as_euler("xyz", degrees=False)
    pose = {
        "position_in_meters": {"x": position[0], "y": position[1], "z": position[2]},
        "rotation_euler_xyz_in_radians": {
            "roll": euler_angles[0],
            "pitch": euler_angles[1],
            "yaw": euler_angles[2],
        },
    }
    return pose


# Get camera extrinsics parameters
T_zed = np.load(  # T_zed_wilson == T_zed_world, because wilson thinks in world frame
    "/home/matt/catkin_ws/src/airo_butler/res/camera_tcps/T_zed_wilson.npy"
)
camera_pose_in_world = tcp2json(T_zed)

# Get camera intrinsics parameters
node = ZED()
node.start_ros()
camera_params = node.zed.camera_params
intrinsics = node.zed.intrinsics_matrix()
camera_intrinsics = {
    "image_resolution": {"width": 1920, "height": 1080},
    "focal_lengths_in_pixels": {"fx": intrinsics[0, 0], "fy": intrinsics[1, 1]},
    "principal_point_in_pixels": {"cx": intrinsics[0, 2], "cy": intrinsics[1, 2]},
}

# Get arm poses
sophie = UR5eClient("sophie")
wilson = UR5eClient("wilson")

arm_left_pose_in_world = tcp2json(wilson.get_tcp_pose())
arm_right_pose_in_world = tcp2json(sophie.get_tcp_pose())


# MAKE RESULTS DIRECTORY

shutil.rmtree(OUT, ignore_errors=True)
makedirs(OUT)

for towel in pbar(listdir(ROOT)):
    for sample in pbar(listdir(towel), desc=towel.name):
        res_folder = OUT / f"sample_{towel.name}_{sample.name}" / "observation_result"
        makedirs(res_folder)

        img = Image.open(sample / "rgb.jpg")
        img.save(res_folder / "image_left.png")

        depth_map = np.load(sample / "depth_map.npy")
        tifffile.imsave(res_folder / "depth_map.tiff", depth_map)

        # make depth image
        depth_image = 255 - cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = Image.fromarray(depth_image.astype(np.uint8))
        depth_image.save(res_folder / "depth_image.jpg")

        with open(res_folder / "camera_intrinsics.json", "w+") as f:
            json.dump(camera_intrinsics, f, indent=4)
        with open(res_folder / "camera_pose_in_world.json", "w+") as f:
            json.dump(camera_pose_in_world, f, indent=4)
        with open(res_folder / "arm_left_tcp_pose_in_world.json", "w+") as f:
            json.dump(arm_left_pose_in_world, f, indent=4)
        with open(res_folder / "arm_right_tcp_pose_in_world.json", "w+") as f:
            json.dump(arm_right_pose_in_world, f, indent=4)
