from copy import deepcopy
import itertools
import json
from pathlib import Path
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from pairo_butler.utils.tools import UGENT, hex_to_rgb, listdir, pbar, pyout


class PointCloudExplorer:
    ROOT = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/experiments/TUM/towel_splatam"
    )

    def __init__(self) -> None:
        self.points, self.rgb, self.alpha, self.scales = self.__load_data()
        self.__apply_masks()
        self.pcd = self.__load_pcd()

    def __load_data(self):
        npz_file = np.load(self.ROOT / "pointcloud.npz")
        return tuple(npz_file[key] for key in npz_file.files)

    def __apply_masks(self, alpha_percentile=75, scale_percentile=25):
        percentile_mask = (
            self.alpha > np.percentile(self.alpha, alpha_percentile)
        ).squeeze(-1)
        scale_mask = (
            self.scales > np.percentile(self.scales, scale_percentile)
        ).squeeze(-1)

        mask = percentile_mask & scale_mask

        self.points = self.points[mask]
        self.rgb = self.rgb[mask]
        self.alpha = self.alpha[mask]
        self.scales = self.scales[mask]

    def __load_pcd(self, up_sample: int = False):
        if up_sample:
            pyout()
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.colors = o3d.utility.Vector3dVector(self.rgb)
        return pcd

    def draw_pointcloud_raw(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.rgb)
        o3d.visualization.draw_geometries([pcd])

    def voxel_downsampling_and_normal_estimation(self, show=False, voxel_size=0.01):
        downpcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )

        if show:
            o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

        return downpcd

    def basic_point_cloud_processing(self, pcd_=None, show=False):
        pcd = deepcopy(pcd_) if pcd_ is not None else deepcopy(self.pcd)

        # convex hull
        hull, _ = pcd.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color(np.array([c / 255 for c in hex_to_rgb(UGENT.BLUE)]))

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
            )

        max_label = labels.max()
        pyout(f"point cloud has {max_label+1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        if show:
            o3d.visualization.draw_geometries([pcd])

        return pcd

    def outlier_removal(self, show=False):
        pcd = deepcopy(self.pcd)
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)

        # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        # self.display_inlier_outlier(voxel_down_pcd, ind)

        cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        self.display_inlier_outlier(voxel_down_pcd, ind)

    def draw(
        self,
        pcd=None,
        camera_tcp=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.5],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ):

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        if pcd is None:
            vis.add_geometry(self.pcd)
        else:
            if isinstance(pcd, list):
                for geom in pcd:
                    vis.add_geometry(geom)
            else:
                vis.add_geometry(pcd)

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

    def display_inlier_outlier(self, cloud, ind, camera_tcp=None):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        # Color the inliers and outliers
        outlier_cloud.paint_uniform_color(hex_to_rgb(UGENT.RED) / 255)
        # inlier_cloud.paint_uniform_color(hex_to_rgb(UGENT.BLUE) / 255)

        # Pass the inliers, outliers, and camera_tcp to the draw function
        self.draw([inlier_cloud, outlier_cloud], camera_tcp=camera_tcp)

    @staticmethod
    def init_states(path):
        states = {}
        with open(path / "pose.txt", "r") as file:
            for line in file:
                if line[0] == "#":
                    continue
                components = line.strip().split()

                timestamp = components[0]
                tx, ty, tz = map(float, components[1:4])
                qx, qy, qz, qw = map(float, components[4:])

                T = np.eye(4)
                T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T[:3, 3] = [tx, ty, tz]

                states[timestamp] = {"tcp": T}
        del tx, ty, tz, qx, qy, qz, qw, line
        return states

    @staticmethod
    def load_pcd_from_rgbd(path, timestamp, state, camera_intrinsics):
        color_raw = o3d.io.read_image(str(path / "rgb" / f"{timestamp}.png"))
        depth_raw = o3d.io.read_image(str(path / "depth" / f"{timestamp}.png"))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        pcd.transform(state["tcp"])
        return pcd

    def depth_images_to_point_clouds(
        self, path: Path, skip=1, distance_threshold=0.2, voxel_size=0.01
    ):
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=720, height=720, fx=920.4, fy=917.4, cx=360.0, cy=359.1
        )
        poses = self.init_states(path)
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-distance_threshold, -distance_threshold, 0.0),
            max_bound=(distance_threshold, distance_threshold, 1.0),
        )

        step = skip
        for timestamp in pbar(list(poses.keys())[::step]):
            pcd = self.load_pcd_from_rgbd(
                path, timestamp, poses[timestamp], camera_intrinsics
            )
            # Downsampling
            pcd = pcd.uniform_down_sample(every_k_points=11)

            # Crop
            pcd = pcd.crop(bounding_box)

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.0)

            yield {"timestamp": timestamp, "tcp": poses[timestamp]["tcp"], "pcd": cl}

            # self.draw(cl, camera_tcp=np.linalg.inv(states[timestamp]["tcp"]))

        return poses

    def draw_registration_result(self, source, target, transformation):
        source_temp = deepcopy(source)
        target_temp = deepcopy(target)
        source_temp.paint_uniform_color(hex_to_rgb(UGENT.GREEN) / 255)
        target_temp.paint_uniform_color(hex_to_rgb(UGENT.LIGHTPURPLE) / 255)
        source_temp.transform(transformation)
        self.draw([source_temp, target_temp])

    def visual_odometry(
        self, states, voxel_size=0.005, sigma=0.1, threshold=1.0, disp_every=115
    ):

        vox = None
        state_tm1 = None
        for ii, state_now in enumerate(states):
            if vox is None:
                vox = state_now["pcd"].voxel_down_sample(voxel_size=voxel_size)
                vox.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.05, max_nn=30
                    )
                )
            else:
                target = state_now["pcd"].voxel_down_sample(voxel_size=voxel_size)
                target.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.05, max_nn=30
                    )
                )
                trans_init = np.eye(4)
                loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
                p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    loss
                )
                # FIRST DO HIDDEN POINT CLOUD REMOVAL

                reg_p2l = o3d.pipelines.registration.registration_icp(
                    vox, target, threshold, trans_init, p2l
                )

                vox = vox.transform(reg_p2l.transformation)

                vox = vox + target
                vox = vox.voxel_down_sample(voxel_size=voxel_size)
                vox, _ = vox.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
                # vox, _ = vox.remove_radius_outlier(nb_points=16, radius=0.05)

                if disp_every > 0 and ii % disp_every == 0:
                    self.draw([vox])

                # self.draw_registration_result(vox, target, reg_p2l.transformation)

                # pyout()

            state_tm1 = state_now


if __name__ == "__main__":
    pce = PointCloudExplorer()
    # downpcd = pce.voxel_downsampling_and_normal_estimation()
    # pce.basic_point_cloud_processing(downpcd, show=True)
    # pce.outlier_removal(show=True)

    states = pce.depth_images_to_point_clouds(
        Path(
            "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS/rgbd_dataset_0"
        )
    )
    pce.visual_odometry(states)
