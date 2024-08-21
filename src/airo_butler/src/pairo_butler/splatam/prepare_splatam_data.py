import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from pairo_butler.utils.tools import listdir, makedirs, pbar, poem, pyout

np.set_printoptions(precision=4, suppress=True)


def get_timestamp(state_T):
    return (
        f"{str(state_T['timestamp']['secs'])}.{str(state_T['timestamp']['nsecs'])[:6]}"
    )


def fix_timestamps(states):
    t = 0
    T = []
    for state_T in states["state_sophie"]:
        if len(T) == 0 or float(get_timestamp(state_T)) > T[-1]:
            T.append(float(get_timestamp(state_T)))
    T = np.array(T)
    dt = np.median(T[1:] - T[:-1])

    for ii, state_T in enumerate(states["state_sophie"]):
        secs = str(int(t)).zfill(10)
        nsecs = str(int((t - int(t)) * 1e9)).zfill(9)
        states["state_sophie"][ii]["timestamp"] = {
            "secs": secs,
            "nsecs": nsecs,
        }
        t += dt
    return states, dt


def cp_rgb_images(states):
    makedirs(ou_folder / "rgb")
    rgb_txt = f"# color images\n# file: 'rgbd_dataset_{in_folder.name}.bag'\n# timestamp filename"
    rgb_npy = np.load(in_folder / "color.npy")

    for ii, (state_T, img_T) in pbar(
        enumerate(zip(states["state_sophie"], rgb_npy)),
        total=rgb_npy.shape[0],
        desc="RGB",
    ):
        img_T = img_T.astype(np.uint8)[..., ::-1]
        timestamp_T = get_timestamp(state_T)
        cv2.imwrite(str(ou_folder / "rgb" / f"{timestamp_T}.png"), img_T)
        rgb_txt += f"\n{timestamp_T} rgb/{timestamp_T}.png"
    with open(ou_folder / "rgb.txt", "w+") as f:
        f.write(rgb_txt)


def cp_depth_images(states):
    makedirs(ou_folder / "depth")
    depth_txt = f"# depth maps\n# file: 'rgbd_dataset_{in_folder.name}.bag'\n# timestamp filename"
    depth_npy = np.load(in_folder / "depth.npy")

    t = -1
    for ii, (state_T, depth_T) in pbar(
        enumerate(zip(states["state_sophie"], depth_npy)),
        total=depth_npy.shape[0],
        desc="3D",
    ):
        depth_T = depth_T.astype(np.uint16)
        

        timestamp_T = get_timestamp(state_T)
        cv2.imwrite(str(ou_folder / "depth" / f"{timestamp_T}.png"), depth_T)
        depth_txt += f"\n{timestamp_T} depth/{timestamp_T}.png"
    with open(ou_folder / "depth.txt", "w+") as f:
        f.write(depth_txt)


def simulate_accelerometer(states):
    T_rs2_sophie = np.load(
        "/home/matt/catkin_ws/src/airo_butler/res/camera_tcps/T_rs2_tcp_sophie.npy"
    )
    acc_txt = f"# accelerometer data\n# file: 'rgbd_dataset_{in_folder.name}.bag'\n# timestamp ax ay az"
    velocities = []
    for t in range(1, len(states["state_sophie"])):
        tcp_sophie_tm1 = np.array(states["state_sophie"][t - 1]["tcp_pose"])
        tcp_sophie_t = np.array(states["state_sophie"][t]["tcp_pose"])

        tcp_rs2_tm1 = T_rs2_sophie @ tcp_sophie_tm1
        tcp_rs2_t = T_rs2_sophie @ tcp_sophie_t

        tcp_tm1_t = np.linalg.inv(tcp_rs2_t) @ tcp_rs2_tm1

        velocities.append(-tcp_tm1_t[:3, 3] / dt)
    velocities = np.stack(velocities, axis=0)
    acceleration = (velocities[1:] - velocities[:-1]) / dt
    acceleration = np.concatenate((np.zeros((2, 3)), acceleration), axis=0)
    acceleration += np.array([0.0, 9.81, 0.0])[None, ...]

    for ii, (state_T, acc_T) in pbar(
        enumerate(zip(states["state_sophie"], acceleration)),
        total=acceleration.shape[0],
    ):
        timestamp_T = get_timestamp(state_T)
        ax, ay, az = acc_T[0], acc_T[1], acc_T[2]
        acc_txt += f"\n{timestamp_T} {ax:.6f} {ay:.6f} {ax:.6f}"
    with open(ou_folder / "accelerometer.txt", "w+") as f:
        f.write(acc_txt)


def convert_pose_to_quat(T):
    """
    Convert a 4x4 homogeneous matrix to translation and quaternion.

    Args:
    - T (numpy.ndarray): 4x4 homogeneous transformation matrix

    Returns:
    - translation (list): [tx, ty, tz]
    - quaternion (list): [qx, qy, qz, qw]
    """
    # Extract translation
    translation = T[:3, 3]

    # Extract rotation matrix
    rotation_matrix = T[:3, :3]

    # Convert rotation matrix to quaternion
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Gives [qx, qy, qz, qw]

    return translation.tolist(), quaternion.tolist()


def extract_poses(states):
    pose_txt = f"# ground truth trajectory\n# file: 'rgbd_dataset_{in_folder.name}.bag'\n# timestamp tx ty tz qx qy qz qw"

    for ii, state_T in enumerate(states["state_sophie"]):
        timestamp_t = get_timestamp(state_T)
        trans_T, quat_T = convert_pose_to_quat(np.array(state_T["tcp_pose"]))

        pose_txt += (
            f"\n{timestamp_t} {trans_T[0]:.4f} {trans_T[1]:.4f} {trans_T[2]:.4f} "
            f"{quat_T[0]:.4f} {quat_T[1]:.4f} {quat_T[2]:.4f} {quat_T[3]:.4f}"
        )
    with open(ou_folder / "pose.txt", "w+") as f:
        f.write(pose_txt)


def intrinsics(states):
    m = np.array(states["rs2_intrinsics"])
    pyout(m)


if __name__ == "__main__":
    IN_ROOT = Path("/media/matt/Expansion/Datasets/towels_depth")
    OU_ROOT = Path("/home/matt/SplaTAM/data/TOWELS")

    for in_folder in (bar := pbar(listdir(IN_ROOT))):
        bar.desc = poem(in_folder.name)

        ou_folder = OU_ROOT / f"rgbd_dataset_{in_folder.name}"

        with open(in_folder / "state.json", "r") as f:
            states = json.load(f)
        states, dt = fix_timestamps(states)
        # intrinsics(states)
        # cp_rgb_images(states)
        cp_depth_images(states)
        simulate_accelerometer(states)
        extract_poses(states)
