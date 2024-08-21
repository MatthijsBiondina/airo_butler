from importlib.machinery import SourceFileLoader
import os
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

from pairo_butler.SplaTAM.viz_scripts.final_recon import load_camera, load_scene_data, make_lineset, render, rgbd2pcd
from pairo_butler.SplaTAM.utils.common_utils import seed_everything
from pairo_butler.utils.tools import pyout


class ReconExplorer:
    def __init__(self):
        pass

    def load_experiment(self):
        experiment_path = "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/configs/tum/towels.py"
        experiment = SourceFileLoader(
            os.path.basename(experiment_path), experiment_path
        ).load_module()
        seed_everything(seed=experiment.config["seed"])

        if "scene_path" not in experiment.config:
            results_dir = os.path.join(
                experiment.config["workdir"], experiment.config["run_name"]
            )
            scene_path = os.path.join(results_dir, "params.npz")
        else:
            scene_path = experiment.config["scene_path"]
        viz_cfg = experiment.config["viz"]

        return scene_path, viz_cfg


    def visualize(self, scene_path, cfg):
        w2c, k = load_camera(cfg, scene_path)
        scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, w2c, k)

        # vis.create_window()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                        height=int(cfg['viz_h'] * cfg['view_scale']),
                        visible=True)

        im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
        init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols
        vis.add_geometry(pcd)

        w = cfg['viz_w']
        h = cfg['viz_h']

        if cfg['visualize_cams']:
            # Initialize Estimated Camera Frustums
            frustum_size = 0.045
            num_t = len(all_w2cs)
            cam_centers = []
            cam_colormap = plt.get_cmap('cool')
            norm_factor = 0.5
            for i_t in range(num_t):
                frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
                frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
                vis.add_geometry(frustum)
                cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
            
            # Initialize Camera Trajectory
            num_lines = [1]
            total_num_lines = num_t - 1
            cols = []
            line_colormap = plt.get_cmap('cool')
            norm_factor = 0.5
            for line_t in range(total_num_lines):
                cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)

        # Initialize View Control
        view_k = k * cfg['view_scale']
        view_k[2, 2] = 1
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        if cfg['offset_first_viz_cam']:
            view_w2c = w2c
            view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
        else:
            view_w2c = w2c
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
        cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

        render_options = vis.get_render_option()
        render_options.point_size = cfg['view_scale']
        render_options.light_on = False

        # Interactive Rendering
        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            if cfg['render_mode'] == 'centers':
                pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
                cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
            else:
                im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
                if cfg['show_sil']:
                    im = (1-sil).repeat(3, 1, 1)
                pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

        # Cleanup
        vis.destroy_window()
        del view_control
        del vis
        del render_options


if __name__ == "__main__":
    recon_explorer = ReconExplorer()
    recon_scene_path, recon_viz_cfg = recon_explorer.load_experiment()
    recon_explorer.visualize(recon_scene_path, recon_viz_cfg)
