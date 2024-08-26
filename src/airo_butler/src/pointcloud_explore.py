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

        self.rmse_buffer = []

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

    def depth_images_to_point_clouds(
        self,
        path: Path,
        skip=1,
        distance_threshold=0.2,
        voxel_size=0.01,
        display_every=25,
        depth_scale_modifier=1.0,
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
                path,
                timestamp,
                poses[timestamp],
                camera_intrinsics,
                depth_scale_modifier,
            )
            # Downsampling
            pcd = pcd.uniform_down_sample(every_k_points=11)

            # Crop
            pcd = pcd.crop(bounding_box)

            cl1, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.0)
            cl2, ind = cl1.remove_radius_outlier(nb_points=10, radius=0.01)

            yield {"timestamp": timestamp, "tcp": poses[timestamp]["tcp"], "pcd": cl2}

        return poses

    def draw_registration_result(self, source, target, transformation):
        source_temp = deepcopy(source)
        target_temp = deepcopy(target)
        source_temp.paint_uniform_color(hex_to_rgb(UGENT.GREEN) / 255)
        target_temp.paint_uniform_color(hex_to_rgb(UGENT.LIGHTPURPLE) / 255)
        source_temp.transform(transformation)
        self.draw([source_temp, target_temp])

    def visual_odometry(
        self,
        states,
        voxel_size=0.005,
        sigma=0.1,
        threshold=1.0,
        # disp_every=115,
        disp_every=0,
        n_keyframes=3,
    ):

        vox = None
        cameras = []
        for ii, state_now in enumerate(states):
            # if not (ii % 115 == 0):
            #     continue

            cameras.append(state_now["tcp"][:3, 3])
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
                camera = state_now["tcp"][:3, 3]
                radius = np.linalg.norm(camera) * 100
                _, pt_map = vox.hidden_point_removal(camera, radius)

                reg_p2l = o3d.pipelines.registration.registration_icp(
                    vox, target, threshold, trans_init, p2l
                )

                vox = vox.transform(reg_p2l.transformation) + target
                vox = vox.voxel_down_sample(voxel_size=voxel_size)
                # vox, _ = vox.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)
                vox, _ = vox.remove_radius_outlier(nb_points=4, radius=voxel_size * 2)

                if disp_every > 0 and ii % disp_every == 0:
                    self.draw([vox])

        self.draw([vox])

    def __preprocess_pcd(self, state_now, voxel_size):
        vox = state_now["pcd"].voxel_down_sample(voxel_size=voxel_size)
        vox.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        return vox

    def __register_next_frame(
        self, sigma, vox, target, threshold, voxel_size, std_ratio=2.0
    ):
        loss_fn = o3d.pipelines.registration.TukeyLoss(k=sigma)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_fn)
        vox = self.compute_normals(vox)
        target = self.compute_normals(target)

        fit = o3d.pipelines.registration.registration_icp(
            vox, target, threshold, np.eye(4), p2l
        )
        self.rmse_buffer.append(
            np.linalg.norm(fit.transformation - np.eye(4), ord="fro")
        )
        vox = vox.transform(fit.transformation) + target
        vox = vox.voxel_down_sample(voxel_size=voxel_size)
        if std_ratio > 0:
            vox, _ = vox.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=std_ratio
            )
        return vox

    def __filter_visibility(self, vox, camera_tcps, min_ratio=0.1, std_ratio=1.0):
        counts = np.zeros((len(vox.points),), dtype=int)
        for camera_tcp in camera_tcps:
            camera = camera_tcp[:3, 3]
            radius = np.linalg.norm(camera) * 100
            _, indexes = vox.hidden_point_removal(camera, radius)
            counts[np.array(indexes)] += 1
        mask = np.flatnonzero(counts >= max(1, min_ratio * len(camera_tcp))).tolist()

        vox = vox.select_by_index(mask)
        if std_ratio > 0:
            vox, _ = vox.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        return vox

    def visual_odometry_pyramid(
        self,
        states,
        voxel_size=0.005,
        sigma=0.1,
        threshold=1.0,
        # disp_every=115,
        disp_every=0,
        n_keyframes=25,
    ):
        vox, camera_tcps = None, []
        for ii, state_now in enumerate(states):
            if ii % n_keyframes == 0 and vox is not None:
                vox = self.__filter_visibility(vox, camera_tcps=camera_tcps)
                # self.draw(vox, camera_tcp=np.linalg.inv(camera_tcps[-1]))
                yield {"vox": vox, "tcps": camera_tcps}
                vox = None

            if vox is None:
                vox = self.__preprocess_pcd(state_now, voxel_size)
                camera_tcps.extend(state_now["tcp"])
            else:
                target = self.__preprocess_pcd(state_now, voxel_size)
                camera_tcps.append(state_now["tcp"])
                vox = self.__register_next_frame(
                    sigma, vox, target, threshold, voxel_size
                )
                # self.draw(vox)

        if vox is not None:
            vox = self.__filter_visibility(vox, camera_tcps=camera_tcps)
            # self.draw(vox, camera_tcp=np.linalg.inv(camera_tcps[-1]))
            yield {"vox": vox, "tcps": camera_tcps}

    def fuse_chunks(self, chunks, voxel_size=0.01, sigma=0.1, threshold=1.0):
        chunk = next(chunks)
        vox = chunk["vox"]
        tcps = chunk["tcps"]
        for chunk in chunks:
            vox = self.__register_next_frame(
                sigma, vox, chunk["vox"], threshold, voxel_size, std_ratio=0.0
            )
            tcps.extend(chunk["tcps"])
            vox = self.__filter_visibility(vox, tcps, min_ratio=0.01, std_ratio=0)
            # self.draw(vox)

        vox = self.__filter_visibility(vox, tcps, min_ratio=0.01, std_ratio=2.0)
        self.draw(vox)

        # vox.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        # )
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     vox, depth=9
        # )
        # bbox = vox.get_axis_aligned_bounding_box()
        # mesh = mesh.crop(bbox)
        # self.draw(mesh)

    def objective_function(self, depth_scale_modifier):
        ROOT = Path(
            "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS"
        )
        rmse = []
        for ii, dataset_path in enumerate(listdir(ROOT)):
            try:
                states = self.depth_images_to_point_clouds(
                    dataset_path, depth_scale_modifier=depth_scale_modifier
                )
                chunks = self.visual_odometry_pyramid(states)
                self.fuse_chunks(chunks)

                rmse.append(np.mean(np.array(self.rmse_buffer) ** 2) ** 0.5)
                self.rmse_buffer = []
            except Exception:
                rmse.append(1.0)
                self.rmse_buffer = []
            if (ii + 1) % 10 == 0:
                break

        return np.median(np.array(rmse))

    @staticmethod
    def compute_normals(pcd, radius=0.1, max_nn=30):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        # pcd.orient_normals_consistent_tangent_plane(
        #     100
        # )  # This helps with the orientation of normals if needed
        return pcd

    def optimize_hyperparameters(
        self,
        population_size=10,
        learning_rate=0.03,
        sigma_learning_rate=0.01,
        num_iterations=100,
    ):
        mu, sigma = 1.0, 0.1

        for iteration in range(num_iterations):
            population = np.random.normal(mu, sigma, population_size)
            population = np.clip(population, a_min=0.5, a_max=1.5)

            fitness = np.array(
                [self.objective_function(individual) for individual in population]
            )

            # Normalize the fitness values
            normalized_fitness = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)

            # Compute the gradient estimate for mu
            gradient_estimate_mu = np.dot(normalized_fitness, population - mu) / (
                population_size * sigma
            )

            # Update the parameter mu
            mu -= learning_rate * gradient_estimate_mu

            # Compute the gradient estimate for sigma (optional but can help)
            gradient_estimate_sigma = np.dot(
                normalized_fitness, (population - mu) ** 2 - sigma**2
            ) / (population_size * sigma)

            # Update sigma with its learning rate
            sigma *= np.exp(sigma_learning_rate * gradient_estimate_sigma)

            # Optionally, print the progress
            pyout(
                f"Iteration {iteration+1}/{num_iterations}, mu = {mu:.4f}, sigma = {sigma:.4f}, fitness = {self.objective_function(mu):.4f}"
            )

            # (mu, sigma, population_size)

        # ROOT = "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS"
        # for dataset_path in listdir(ROOT):

        # states =


if __name__ == "__main__":
    root_in = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS_raw/"
    )
    root_ou = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS/"
    )

    # for dataset_path in listdir(root_in):
    #     pyout()

    pce = PointCloudExplorer()
    # pce.optimize_hyperparameters()
    states = pce.depth_images_to_point_clouds(
        Path(
            "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS_pre/rgbd_dataset_0"
        ),
        depth_scale_modifier=0.94,
    )
    chunks = pce.visual_odometry_pyramid(states)

    # # chunk1 = next(chunks)
    # # chunk2 = next(chunks)
    # # chunk1['vox'].paint_uniform_color(hex_to_rgb(UGENT.PINK) / 255)
    # # chunk2['vox'].paint_uniform_color(hex_to_rgb(UGENT.GREEN) / 255)

    # # pce.draw([chunk1['vox'], chunk2['vox']])

    for chunk in chunks:
        pce.draw(chunk["vox"], camera_tcp=np.linalg.inv(chunk["tcps"][0]))

    # pce.fuse_chunks(chunks)
