import itertools
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from pairo_butler.utils.tools import listdir, makedirs, pbar, pyout
from scipy.spatial.transform import Rotation as R
import cv2


def evenly_spaced_indexes(nr_of_indexes, total):
    indexes = list(
        set(
            np.linspace(
                0,
                total - 1,
                nr_of_indexes,
                dtype=int,
            ).tolist()
        )
    )
    return indexes


def draw_pcd(
    pcd=None,
    camera_tcp=np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    include_origin=True,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud or geometries
    if isinstance(pcd, list):
        for geom in pcd:
            vis.add_geometry(geom)
    else:
        vis.add_geometry(pcd)

    # Add coordinate frame at the origin if include_origin is True
    if include_origin:
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(origin_frame)

    # Set the camera to the TCP if provided
    if camera_tcp is not None:
        ctr = vis.get_view_control()

        # Get current camera parameters
        camera_params = ctr.convert_to_pinhole_camera_parameters()

        # Set the extrinsic matrix to the provided TCP (camera pose)
        camera_params.extrinsic = camera_tcp

        # Apply the updated parameters
        ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.run()
    vis.destroy_window()


class State:
    def __init__(
        self,
        timestamp: float,
        color_path: Path,
        depth_path: Path,
        intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=720, height=720, fx=920.4, fy=917.4, cx=360.0, cy=359.1
        ),
        extrinsics: np.ndarray | None = None,
    ) -> None:
        self.timestamp: float = timestamp
        self.intrinsics: o3d.camera.PinholeCameraIntrinsic = intrinsics
        self.extrinsics: np.ndarray = extrinsics
        self.color_path: Path = color_path
        self.depth_path: Path = depth_path
        self.color_img = None
        self.depth_img = None


class TrajectoryMapper:
    def __init__(self, root: Path):
        self.root = root
        self.states = self.__load_trial_data()

        # Hyperparameters
        self.hyperparameter_depth_scale = 1000.0 / 0.94

    def __load_trial_data(self):
        states = []

        with open(self.root / "pose.txt", "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                components = line.strip().split()

                timestamp = components[0]
                tx, ty, tz = map(float, components[1:4])
                qx, qy, qz, qw = map(float, components[4:])

                extrinsics = np.eye(4)
                extrinsics[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                extrinsics[:3, 3] = [tx, ty, tz]
                extrinsics = np.linalg.inv(extrinsics)

                color_path = self.root / "rgb" / f"{timestamp}.png"
                depth_path = self.root / "depth" / f"{timestamp}.png"

                if os.path.exists(color_path) and os.path.exists(depth_path):
                    states.append(
                        State(
                            timestamp=float(timestamp),
                            color_path=color_path,
                            depth_path=depth_path,
                            extrinsics=extrinsics,
                        )
                    )

        return states

    def __load_state_pointcloud(
        self,
        state: State,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False,
        distance_threshold=0.25,
    ):
        depth_image = o3d.io.read_image(str(state.depth_path))
        color_image = o3d.io.read_image(str(state.color_path))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            self.hyperparameter_depth_scale,
            depth_trunc,
            convert_rgb_to_intensity,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, state.intrinsics, state.extrinsics
        )
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-distance_threshold, -distance_threshold, 0.0),
            max_bound=(distance_threshold, distance_threshold, 1.0),
        )
        pcd = pcd.crop(bounding_box)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return pcd

    def reproject(self, pcd_legacy, state):
        pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_legacy)
        rgbd = pcd.project_to_rgbd_image(
            720,
            720,
            o3d.core.Tensor(state.intrinsics.intrinsic_matrix, o3d.core.Dtype.Float32),
            o3d.core.Tensor(state.extrinsics, o3d.core.Dtype.Float32),
            depth_scale=1000.0,
            depth_max=4000.0,
        )

        color_image = rgbd.color
        depth_image = rgbd.depth

        # Convert the images to numpy arrays for display
        color_np = (np.array(color_image.to_legacy()) * 255).astype(np.uint8)
        depth_np = np.array(depth_image.to_legacy()).astype(np.uint16)

        return color_np, depth_np

    def preprocess_depth_images(self):
        for state in pbar(self.states, desc="Reprojection"):
            pcd = self.__load_state_pointcloud(state)
            color, depth = self.reproject(pcd, state)
            state.color_img = color
            state.depth_img = depth

    @staticmethod
    def load_pcd_from_rgbd(
        path, timestamp, state, camera_intrinsics, depth_scale_modifier=1.0
    ):
        color_raw = o3d.io.read_image(str(path / "rgb" / f"{timestamp}.png"))
        depth_raw = o3d.io.read_image(str(path / "depth" / f"{timestamp}.png"))
        depth_raw_np = np.asarray(depth_raw)
        # Multiply the depth image by the scale modifier
        scaled_depth_np = depth_raw_np * depth_scale_modifier

        # Convert the scaled depth numpy array back to an Open3D image
        scaled_depth_raw = o3d.geometry.Image(
            scaled_depth_np.astype(np.uint16)
        )  # Use uint16 if depth is typically 16-bit

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, scaled_depth_raw, convert_rgb_to_intensity=True
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        pcd.transform(state["tcp"])
        return pcd


if __name__ == "__main__":
    # CONSTANTS

    root_in = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS_raw/"
    )
    root_ou = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS_pre/"
    )
    shutil.rmtree(root_ou, ignore_errors=True)
    makedirs(root_ou)

    for dataset_path_in in pbar(listdir(root_in)):
        dataset_path_ou = Path(str(dataset_path_in).replace("TOWELS_raw", "TOWELS_pre"))
        tm = TrajectoryMapper(dataset_path_in)
        tm.preprocess_depth_images()

        shutil.copytree(dataset_path_in, dataset_path_ou)

        for state in tm.states:
            cv2.imwrite(
                str(dataset_path_ou / "depth" / f"{state.timestamp}.png"), state.depth_img
            )

        pyout()
