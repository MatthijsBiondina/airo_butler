import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import cv2
import numpy as np
import rospkg
import yaml
from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.utils.tools import UGENT, listdir, load_mp4_video, pbar, pyout
import rospy as ros

np.set_printoptions(precision=2, suppress=True)


class VisibilityChecker:
    """
    This class processes video and keypoint data to classify keypoints as visible,
    obscured, or out-of-bounds based on their position relative to the camera view and
    other measured keypoints.

    Returns:
        _type_: VisibilityChecker
    """

    RATE = 60

    def __init__(self, name: str = "visibility_checker") -> None:
        """
        Initializes the VisibilityChecker class by setting up the ROS node name,
        loading configuration settings from a YAML file, and loading the transformation
        matrix from a NumPy file. This transformation matrix is specifically for
        converting coordinates from the robot arm's Tool Center Point (TCP) to the
        camera's TCP mounted on the wrist, effectively transforming 3D points from
        the robot's frame of reference to the camera's frame of reference.

        Args:
            name: The name of the ROS node. This name is used for ROS logging and
                diagnostics. Defaults to "visibility_checker".
        """
        # Set the name of the ROS node
        self.node_name: str = name

        # Load the configuration settings from 'labelling_config.yaml'
        # The configuration file contains settings such as folder paths, camera
        # parameters, and other necessary data for processing
        config_path: Path = Path(__file__).parent / "labelling_config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Load the 3D to 3D transformation matrix for converting coordinates from
        # the robot arm's TCP to the camera's TCP mounted on the wrist. This matrix
        # is crucial for accurately projecting 3D keypoints onto the camera's 2D image
        # plane by first transforming them into the camera's 3D coordinate frame.
        matrix_path: Path = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "res"
            / "camera_tcps"
            / "T_rs2_tcp_sophie.npy"
        )
        self.T_sophie_cam: np.ndarray = np.load(matrix_path)

    def start_ros(self) -> None:
        """
        Initializes the ROS node with the specified node name, sets the logging level to INFO,
        and establishes a rate for the node's execution cycle. Additionally, it configures a
        shutdown hook to ensure OpenCV windows are closed gracefully upon the node's shutdown.

        This method should be called after initializing the class and before entering the main
        processing loop to ensure that the ROS node is properly set up and that ROS-specific
        functionalities (like publishing or subscribing to topics) can be used.
        """
        # Initialize the ROS node with the specified name and set the logging level to INFO
        ros.init_node(self.node_name, log_level=ros.INFO)

        # Set the rate at which to run the node; this facilitates a consistent cycle time
        # in the node's main loop
        self.rate = ros.Rate(
            self.RATE
        )  # RATE is defined in the class and represents the desired frequency

        # Register a callback function that will be called upon node shutdown;
        # this ensures that OpenCV windows are closed gracefully
        ros.on_shutdown(cv2.destroyAllWindows)

        # Log a confirmation message indicating that the ROS node has been successfully initialized
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self) -> None:
        """
        Main execution loop of the VisibilityChecker node. This function iterates through each
        video trial specified in the configuration directory, checking for ROS node shutdown
        signals, loading trial data, projecting keypoints onto camera frames, and determining
        their visibility status. For each valid trial, it updates the trial data with visibility
        information and saves the updated data.

        It uses a progress bar (pbar) to visually track the processing of trials, enhancing
        user feedback during operation. Trials deemed invalid are skipped, ensuring that only
        trials with the required data are processed.
        """
        # Iterate through each trial in the specified folder, with a progress bar for feedback
        for ii, trial in pbar(
            enumerate(listdir(self.config["folder"])), desc="Determining Visibility"
        ):
            ros.loginfo(trial)

            # Check for a ROS shutdown signal to gracefully exit if needed
            if ros.is_shutdown():
                break

            # Load the trial data and check its validity
            data, valid = VisibilityChecker.load_trial_data(trial)
            # Skip processing for invalid trials
            if not valid:
                continue

            # Project keypoints onto camera frames for the current trial
            data = self.__project_keypoints_onto_camera_frames(data)

            # Check whether keypoints are visible
            data = self.determine_keypoints_visibility(data)

            # Determine and plot visibility status for keypoints in the trial
            VisibilityChecker.plot_visible_and_obscured_keypoints(data)

            # Save the updated trial data with visibility information
            VisibilityChecker.save_data_with_visibility_labels(trial, data)

    @staticmethod
    def load_trial_data(path: Path) -> Tuple[Dict[str, Any], bool]:
        """
        Loads the trial data for visibility checking from a specified directory path. This method
        reads the trial's state from a JSON file and loads video frames from an MP4 video file located
        within the same directory. It assesses the trial's validity based on a flag within the JSON
        state file and returns both the trial data and its validity status.

        Args:
            path (Path): The file system path to the directory containing the trial's # Extract the 2D projected keypoints for the current framedata files.
                        This should include 'state.json' for metadata and 'video.mp4' for video frames.

        Returns:
            Tuple[Dict[str, Any], bool]: A tuple containing the trial's data as a dictionary and a
                                        boolean flag indicating the trial's validity. The data dictionary
                                        includes loaded video frames and any other relevant information
                                        extracted from the 'state.json' file.
        """
        # Load the trial's state from the 'state.json' file
        # raise NotImplementedError

        try:
            with open(path / "state.json", "r") as f:
                state: Dict[str, Any] = json.load(f)

            # Load the video frames from the 'video.mp4' file located in the same directory
            state["frames"] = load_mp4_video(path / "video.mp4")

            # Extract and return the trial's validity flag along with the loaded state data
            # 'valid' is expected to be a key in the state dict indicating if the trial data
            # is complete and usable for visibility checking
            return state, state.get(
                "valid", False
            )  # Provide a default as False in case 'valid' key is missing
        except json.decoder.JSONDecodeError:
            ros.logwarn(f"Could not decode {path}.")
            return None, False

    def __project_keypoints_onto_camera_frames(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Projects 3D keypoints onto 2D camera frames based on the camera's transformation matrix
        and intrinsic parameters. This method updates the input data dictionary with a new key
        'keypoints_clean' containing the projected keypoints for each frame.

        Args:
            data (Dict[str, Any]): A dictionary containing the trial's data, including 3D keypoints
                                ('keypoints_world'), the camera's transformation matrix for each frame
                                ('state_sophie' -> 'tcp_pose'), and camera intrinsic parameters
                                ('rs2_intrinsics').

        Returns:
            Dict[str, Any]: The updated dictionary with an added 'keypoints_clean' key, which contains
                            the 2D projections of the 3D keypoints for each frame. The projection is
                            computed using the camera's transformation matrix and intrinsic parameters.

        Note:
            This method mutates the input dictionary by adding the 'keypoints_clean' key.
        """
        # Determine the number of frames in the data
        nr_of_frames: int = len(data["keypoints"])
        # Initialize an empty list for each frame to store clean keypoints
        data["keypoints_clean"] = [[] for _ in range(nr_of_frames)]

        # Get the number of keypoints to be processed
        nr_of_keypoints: int = len(data["keypoints_world"])

        # Iterate through each keypoint and each frame
        for kp_idx in range(nr_of_keypoints):
            for frame_idx in range(nr_of_frames):
                # Compute the camera pose in the robot's coordinate frame
                camera_tcp: np.ndarray = (
                    np.array(data["state_sophie"][frame_idx]["tcp_pose"])
                    @ self.T_sophie_cam
                )
                # Load the camera intrinsic parameters
                intrinsics_matrix: np.ndarray = np.array(data["rs2_intrinsics"])
                # Reshape keypoint mean for compatibility with transformation calculations
                keypoint_mean: np.ndarray = np.array(
                    data["keypoints_world"][kp_idx]["mean"]
                )[:, None]

                # Project the 3D keypoint onto the 2D camera frame
                keypoint_projection: np.ndarray = (
                    KalmanFilter.calculate_expected_measurements(
                        keypoint_world=keypoint_mean,
                        camera_tcp=camera_tcp,
                        camera_intrinsics=intrinsics_matrix,
                    )
                )

                # Append the projected keypoint to the clean keypoints list for the frame
                data["keypoints_clean"][frame_idx].append(keypoint_projection)
                # Set the visibility flag for the keypoint (0.0 indicates out-of-bounds or not visible)
                data["keypoints_clean"][frame_idx][kp_idx][2] = 0.0

        return data

    def determine_keypoints_visibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the visibility of keypoints in each frame and updates the data dictionary with
        the visibility information. This method assesses whether each keypoint is visible, obscured,
        or out-of-bounds by comparing the projected keypoints ('keypoints_clean') with the actual
        measured keypoints ('keypoints') in the data.

        Args:
            data (Dict[str, Any]): The trial data containing both projected ('keypoints_clean')
                                and measured ('keypoints') keypoints for each frame.

        Returns:
            Dict[str, Any]: The updated trial data with visibility information appended to each
                            keypoint in 'keypoints_clean'. The visibility information is added as
                            a third element to each keypoint array.

        Note:
            This method directly modifies the input dictionary by updating the 'keypoints_clean'
            key with visibility information for each keypoint.
        """
        # Determine the number of frames to process
        nr_of_frames: int = len(data["keypoints"])

        # Iterate through each frame to determine the visibility of keypoints
        for frame_idx in range(nr_of_frames):
            # Extract the 2D projected keypoints for the current frame
            keypoints_clean: np.ndarray = np.array(data["keypoints_clean"][frame_idx])
            if keypoints_clean.size == 0:
                continue
            keypoints_clean = keypoints_clean[:, :2]

            # Check if there are measured keypoints for the current frame, and convert them to numpy array if present
            keypoints_measured: Optional[np.ndarray] = None
            if data["keypoints"][frame_idx] is not None:
                keypoints_measured = np.array(
                    data["keypoints"][frame_idx]
                )  # Extract the 2D projected keypoints for the current frame
            visibility: np.ndarray = self.__determine_visibility(
                keypoint_clean=keypoints_clean, keypoints_measured=keypoints_measured
            )

            # Concatenate the visibility information with the clean keypoints
            keypoints_clean: np.ndarray = np.concatenate(
                (keypoints_clean, visibility), axis=1
            )

            # Update the 'keypoints_clean' data with the new information including visibility
            data["keypoints_clean"][frame_idx] = keypoints_clean.tolist()

        return data

    def __determine_visibility(
        self, keypoint_clean: np.ndarray, keypoints_measured: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Determines the visibility of each projected keypoint by evaluating its position
        relative to the camera's frame boundaries and its closeness to actual measured
        keypoints. This method classifies each keypoint into one of three visibility
        categories: out-of-bounds, obscured, or visible, represented by 0.0, 1.0, and
        2.0, respectively.

        Args:
            keypoint_clean (np.ndarray): The 2D coordinates of keypoints projected onto
                                         the camera frame, with shape (N, 2, 1), where N
                                         is the number of keypoints.
            keypoints_measured (Optional[np.ndarray]):
                The 2D coordinates of keypoints detected in the camera frame, with shape
                (M, 2), where M is the number of detected keypoints. This parameter may
                be None if no keypoints are detected.

        Returns:
            np.ndarray: An array of visibility scores for each keypoint in `keypoint_clean`,
                        with shape (N, 1, 1). Each score is one of the following values:
                        - 0.0 indicates the keypoint is out-of-bounds of the camera frame.
                        - 1.0 indicates the keypoint is within the camera frame but obscured.
                        - 2.0 indicates the keypoint is visible and unobscured within
                          the camera frame.

        The visibility determination process involves two main steps:
        1. Checking if keypoints are within the bounds of the camera frame, based on the
           configured frame dimensions (`self.config["frame_width"]` and
           `self.config["frame_height"]`).
        2. For keypoints within the frame, further checking if they are close to any measured
           keypoints to determine if they are unobscured (visible) or not.
        """
        visibility = np.zeros((keypoint_clean.shape[0], 1))

        # Determine if keypoints are within the camera frame bounds
        width_mask = (0 < keypoint_clean[:, 0, :]) & (
            keypoint_clean[:, 0, :] < self.config["frame_width"]
        )
        height_mask = (0 < keypoint_clean[:, 1, :]) & (
            keypoint_clean[:, 1, :] < self.config["frame_height"]
        )
        visibility[width_mask & height_mask] = 1.0  # Mark as obscured if within bounds

        # If keypoints are measured, determine the proximity to clean keypoints to assess visibility
        if keypoints_measured is not None and keypoints_measured.size > 0:
            distance = np.linalg.norm(
                keypoint_clean.squeeze(-1)[:, None, :] - keypoints_measured[None, :, :],
                axis=-1,
            )
            closest_measured_point = np.argmin(distance, axis=0)
            # We assume that the cleaned keypoint corresponds with the closest measured keypoint.
            # So if it is not closest to any of the measured keypoints, then it must be obscured.
            unobscured = np.any(
                np.arange(keypoint_clean.shape[0])[:, None]
                == closest_measured_point[None, :],
                axis=1,
            )
            visibility[(visibility == 1.0) & unobscured[:, None]] = (
                2.0  # Mark as visible if unobscured
            )

        return visibility[:, None, :]

    @staticmethod
    def plot_visible_and_obscured_keypoints(data: Dict[str, Any]) -> None:
        """
        Visualizes and displays each frame of the video with keypoints overlaid, color-coded by their visibility status.
        Keypoints classified as obscured are marked in red, while visible keypoints are marked in green.
        Keypoints out-of-bounds (0.0) are not displayed. The video frames are shown in sequence,
        and the display can be exited by pressing the 'q' key.

        Args:
            data (Dict[str, Any]): A dictionary containing the processed data of a trial, including 'frames'
                                (a list of images) and 'keypoints_clean' (the corresponding keypoints
                                with visibility information for each frame).

        Note:
            - This function directly manipulates the frames by drawing circles on them according to the keypoints'
            visibility statuses.
            - The 'frames' list is expected to contain images in a format compatible with OpenCV (usually BGR numpy arrays).
            - The 'keypoints_clean' list contains for each frame a list of keypoints, where each keypoint
            is represented as a list with the x and y coordinates followed by a visibility status.
        """
        for idx in range(len(data["frames"])):
            frame = data["frames"][idx]  # Access the current frame
            keypoints = data["keypoints_clean"][
                idx
            ]  # Access keypoints for the current frame

            for keypoint in keypoints:
                kp_x, kp_y = (
                    keypoint[0][0],
                    keypoint[1][0],
                )  # Extract keypoint coordinates

                # Check the visibility status and draw keypoints accordingly
                if keypoint[2][0] == 1.0:  # Obscured keypoints
                    frame = VisibilityChecker.draw_colored_point_on_image(
                        frame, (kp_x, kp_y), UGENT.RED
                    )
                elif keypoint[2][0] == 2.0:  # Visible keypoints
                    frame = VisibilityChecker.draw_colored_point_on_image(
                        frame, (kp_x, kp_y), UGENT.GREEN
                    )
                # Note: Keypoints with visibility status 0.0 (out-of-bounds) are not drawn

            # Display the current frame with keypoints overlaid
            cv2.imshow(
                "Visibility", np.array(frame)[..., ::-1]
            )  # Convert frame for correct color display if needed
            # Break and exit if 'q' key is pressed
            if cv2.waitKey(100) & 0xFF == ord("q"):
                ros.loginfo("Process interrupted by user.")
                ros.signal_shutdown("Interrupted by user")
                sys.exit(0)

    @staticmethod
    def draw_colored_point_on_image(
        image: Image.Image, coord: Tuple[int, int], color: str, radius: int = 5
    ) -> Image.Image:
        """
        Draws a colored circle around a specified coordinate on an image to visually indicate a keypoint.
        This function is primarily used to mark keypoints with different colors based on their visibility status.

        Args:
            image (Image.Image): The PIL Image object on which the keypoint indicator will be drawn.
            coord (Tuple[int, int]): The (x, y) coordinates of the keypoint on the image.
            color (str): The color of the outer circle to indicate the keypoint's visibility status.
                        The color must be specified in a format recognized by PIL (e.g., 'red', 'green').
            radius (int, optional): The radius of the circle to be drawn around the keypoint. Defaults to 5 pixels.

        Returns:
            Image.Image: The modified PIL Image object with the keypoint indicator drawn on it.

        Note:
            This function modifies the input image in place but also returns the modified image for convenience.
            A blue outline is always drawn as a base with the specified color outline drawn slightly inside it
            to create a two-tone effect, enhancing visibility against varying backgrounds.
        """
        draw = ImageDraw.Draw(image)  # Create a drawing context
        # Draw a blue circle as the base outline around the keypoint
        draw.ellipse(
            [
                (int(round(coord[0] - radius)), int(round(coord[1] - radius))),
                (int(round(coord[0] + radius)), int(round(coord[1] + radius))),
            ],
            outline=UGENT.BLUE,
            width=2,
        )
        # Draw the colored circle specified by the 'color' argument inside the blue circle
        draw.ellipse(
            [
                (
                    int(round(coord[0] - (radius - 1))),
                    int(round(coord[1] - (radius - 1))),
                ),
                (
                    int(round(coord[0] + (radius - 1))),
                    int(round(coord[1] + (radius - 1))),
                ),
            ],
            outline=color,
            width=2,
        )

        return image

    @staticmethod
    def save_data_with_visibility_labels(path: Path, data: Dict[str, Any]) -> None:
        """
        Saves the trial data, which now includes visibility information for each keypoint, to a JSON file.
        This function is designed to serialize the updated data dictionary (excluding video frames for size efficiency)
        and save it in a structured JSON format, facilitating further analysis or visualization.

        Args:
            path (Path): The directory path where the JSON file will be saved. The function will overwrite
                        'state.json' in this directory with the updated data.
            data (Dict[str, Any]): The data dictionary containing trial information, including keypoints'
                                visibility statuses. Note that the 'frames' key, if present, will be
                                removed before saving to minimize file size.

        Note:
            This function modifies the input data by removing the 'frames' key to avoid saving image data
            in the JSON file. Ensure that this operation is acceptable in your workflow before calling
            the function or clone the data dictionary if necessary.
        """
        # Remove the 'frames' key to exclude video frames from being saved (to save space)

        if "frames" in data:
            del data["frames"]

        # Save the updated trial data (without frames) to 'state.json' in the specified path
        with open(path / "state.json", "w") as f:
            json.dump(
                data, f, indent=2
            )  # Use an indentation level of 2 for readability


def main():
    node = VisibilityChecker()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
