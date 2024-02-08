import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import cv2
import numpy as np
import pyautogui
import yaml
from PIL import Image, ImageDraw
from pairo_butler.utils.custom_exceptions import BreakException
from pairo_butler.plotting.pygame_plotter import PygameWindow
from pairo_butler.utils.tools import UGENT, listdir, pbar, pyout
import rospy as ros


class KeypointLabeler:
    """
    A class designed to label keypoints in videos, featuring methods for loading data,
    handling user input, and saving the labeled keypoints.
    """

    RATE = 60  # The frequency at which the ROS node operates.

    # Parameters for Lucas-Kanade optical flow calculation, used in tracking keypoints
    # between video frames.
    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    def __init__(self, name: str = "keypoint_labeler"):
        """
        Initializes the ROS node with a specified name and loads configuration settings
        from a YAML file.

        Args:
            name (str, optional): Name of the ros node. Defaults to "keypoint_labeler".
        """
        # Store the name of the ROS node.
        self.node_name = name

        # Open the YAML configuration file located in the same directory as this script.
        # Path(__file__) refers to the path of the current script.
        # .parent gets the directory containing the script.
        # "labelling_config.yaml" is the name of the config file to be loaded.
        with open(Path(__file__).parent / "labelling_config.yaml", "r") as f:
            # Load the YAML file content into the self.config dictionary.
            # yaml.safe_load(f) reads and parses the YAML file, ensuring that it's loaded
            # in a safe manner to prevent execution of arbitrary code.
            self.config = yaml.safe_load(f)

    def start_ros(self):
        """
        Initializes the ROS node and sets up a ROS rate for timing.
        """
        # Initialize the ROS node with the specified name. The log_level is set to INFO,
        # meaning that messages at this level and above will be logged (e.g., INFO, WARN, ERROR, FATAL).
        ros.init_node(self.node_name, log_level=ros.INFO)

        # Set up a ROS Rate object. This is used to regulate the frequency of the loop execution.
        # The rate is determined by the RATE class attribute, which specifies the desired loops per second.
        self.rate = ros.Rate(self.RATE)

        # Register shutdown function to close cv2 windows
        ros.on_shutdown(cv2.destroyAllWindows)

        # Log a message at the INFO level indicating that the node has been successfully initialized.
        # This is useful for debugging and monitoring the node's status.
        ros.loginfo(f"{self.node_name}: OK!")

    def run(self):
        """
        Main loop that processes each video file in the specified directory, loading the
        trial data, labeling it, and handling user interactions.
        """
        # Iterate through each trial in the directory specified in the config file.
        # `pbar` is a progress bar function that wraps the iterable returned by `listdir`,
        # providing visual feedback in the console about the progress of the labeling process.
        # `listdir(self.config["folder"])` lists all items in the directory specified by the 'folder' key in the config dictionary.
        for trial in pbar(listdir(self.config["folder"]), desc="Labelling"):
            # Load the trial data. `__load_data_trial` returns the video data (vid) and a flag indicating
            # whether it has already been labeled (done).
            vid, done = self.__load_data_trial(trial)

            # If the configuration specifies to only label unlabeled files (`only_unlabeled_files` is True)
            # and the current file is already labeled (`done` is True), skip to the next iteration.
            if self.config["only_unlabeled_trials"] and done:
                continue

            # Label the data for the current trial. The `__label_data` method updates `vid` with the labeled keypoints
            # and potentially other modifications made during the labeling process.
            vid = self.__label_data(vid, trial)

    def __load_data_trial(self, path: Path):
        """
        Loads the state information and video frames for a given trial.

        Args:
            path (Path): The directory path where the trial's data is located.

        Returns:
            Tuple[Dict[str, Any], bool]: A tuple containing the trial's state information as a dictionary
            and a boolean indicating whether the trial has already been labeled.
        """
        # Open and read the state.json file, which contains metadata and possibly keypoints for the trial.
        # This file is expected to be in the same directory as the trial's video file.
        with open(path / "state.json", "r") as f:
            state = json.load(f)

        # Initialize an empty list to store frames from the video.
        state["frames"] = []

        # Open the video file associated with the trial using OpenCV.
        cap = cv2.VideoCapture(str(path / "video.mp4"))

        while True:
            # Read the next frame from the video.
            success, frame = cap.read()

            # If reading the frame was unsuccessful, break out of the loop (end of video).
            if not success:
                break

            # Convert the BGR frame (OpenCV's default color format) to RGB and append it to the `frames` list.
            # This conversion is necessary because PIL uses RGB format, and we're converting OpenCV frames to PIL images.
            state["frames"].append(Image.fromarray(frame[..., ::-1]))

        # Assume the trial has been labeled unless proven otherwise.
        labeled = True
        try:
            # Attempt to access the `keypoints` in the state. If this fails (KeyError), it means
            # the keypoints have not been labeled for this trial.
            state["keypoints"]
        except KeyError:
            labeled = False
            # Initialize the `keypoints` list with None, one for each frame, indicating no keypoints.
            state["keypoints"] = [None] * len(state["frames"])

        try:
            # Check if the trial is marked as valid. If the `valid` key doesn't exist, assume it's valid.
            state["valid"]
        except KeyError:
            labeled = False
            state["valid"] = True

        # Assert that the length of `state_sophie` matches the number of frames.
        # This is a check to ensure each frame has corresponding state information.
        assert len(state["state_sophie"]) == len(state["frames"])

        # Return the state dictionary and the labeled flag.
        return state, labeled

    def __label_data(self, vid: Dict[str, Any], path):
        """
        Manages the labeling interface, allowing users to navigate through video frames and
        label keypoints via mouse clicks. It uses optical flow to extrapolate keypoints
        across frames.

        Args:
            vid (Dict[str, Any]): Dictionary containing video data, including frames and keypoints.
            path (Path): The file path of the current video being labeled, used for saving.
        """

        # Inner function to handle mouse events (clicks) on the OpenCV window.
        # This allows users to add, remove, or modify keypoints by clicking on the video frames.
        def mouse_event(event, x, y, flags, params):
            nonlocal vid, idx  # Use variables from the outer scope.
            # Process the mouse click to update keypoints and the current frame index.
            vid["keypoints"], idx = self.__process_mouse(
                event, (x, y), vid["keypoints"], idx
            )

        # Create an OpenCV window with the name of the ROS node.
        cv2.namedWindow(self.node_name)
        # Assign the mouse event handling function to the created window.
        cv2.setMouseCallback(self.node_name, mouse_event)

        idx = 0  # Start with the first frame.
        while True:
            # Ensure the current index stays within the bounds of the video frames list.
            idx = np.clip(idx, 0, len(vid["frames"]) - 1)

            # Attempt to use optical flow to extrapolate keypoints for the current frame.
            vid = self.__extrapolate_keypoints(vid, idx)

            # Draw the current keypoints on the copied current frame image.
            img = self.__draw_keypoints_on_image(
                vid["frames"][idx].copy(), points=vid["keypoints"][idx]
            )

            # If the trial is marked as invalid, convert the image to grayscale.
            if not vid["valid"]:
                img = img.convert("L")
                cv2.imshow(self.node_name, np.array(img))
            else:
                # For valid trials, display the image in its original color (RGB to BGR for OpenCV).
                cv2.imshow(self.node_name, np.array(img)[..., ::-1])

            # Wait for a key press and process it to navigate frames, save data, or exit.
            key = cv2.waitKey(0) & 0xFF
            try:
                idx = self.__process_keyboard(key, idx, vid, path)
            except BreakException:
                # Break from the loop if a BreakException is raised (e.g., pressing the ESC key).
                break

            # Sleep for a brief period to maintain a consistent rate of processing.
            self.rate.sleep()

        # Close the OpenCV window once the loop is exited.
        cv2.destroyAllWindows()

    def __extrapolate_keypoints(self, vid: Dict[str, Any], idx):
        """
        Uses optical flow to predict the position of keypoints in the next frame based on
        their positions in the current frame.

        Args:
            vid (Dict[str, Any]): The video data structure containing frames and keypoints.
            idx (int): The index of the current frame for which keypoints need to be extrapolated.

        Returns:
            Dict[str, Any]: The updated video data structure with extrapolated keypoints for the current frame.
        """
        # Skip extrapolation if there are no previous keypoints to base the prediction on,
        # if this is the first frame, or if the previous frame has no keypoints.
        if (
            vid["keypoints"][idx] is not None
            or idx == 0
            or vid["keypoints"][idx - 1] is None
            or len(vid["keypoints"][idx - 1]) == 0
        ):
            return vid

        # Prepare the previous keypoints for optical flow calculation, ensuring the correct shape and type.
        pts_prev = vid["keypoints"][idx - 1][:, None, :].astype(np.float32)
        # Convert previous and next frames to grayscale, which is required for calcOpticalFlowPyrLK.
        img_prev_gray = np.array(vid["frames"][idx - 1].convert("L"))
        img_next_gray = np.array(vid["frames"][idx].convert("L"))

        # Calculate the optical flow to predict the new positions of the keypoints in the next frame.
        extrapolated_points, _, _ = cv2.calcOpticalFlowPyrLK(
            img_prev_gray,
            img_next_gray,
            pts_prev,
            None,
            **self.LK_PARAMS,  # Parameters for the Lucas-Kanade method.
        )
        # Reshape the extrapolated points for consistency.
        extrapolated_points = extrapolated_points.reshape(-1, 2)

        # Remove any keypoints that fall outside the frame boundaries.
        for ii in range(len(extrapolated_points) - 1, -1, -1):
            kp = extrapolated_points[ii]
            if not (0 < kp[0] < vid["frames"][idx].width) or not (
                0 < kp[1] < vid["frames"][idx].height
            ):
                extrapolated_points = np.concatenate(
                    (extrapolated_points[:ii], extrapolated_points[ii + 1 :]), axis=0
                )

        # Update the keypoints for the current frame with the extrapolated points.
        vid["keypoints"][idx] = extrapolated_points.reshape(-1, 2)
        return vid

    def __draw_keypoints_on_image(
        self, image: Image, points: Union[None, List[Tuple[int, int]]], radius: int = 5
    ):
        """
        Visualizes keypoints on a frame by drawing circles around each keypoint.

        Args:
            image (Image): The PIL image object on which to draw the keypoints.
            points (Union[None, List[Tuple[int, int]]]): A list of tuples where each tuple contains the (x, y) coordinates
                of a keypoint. If None, no keypoints are drawn.
            radius (int, optional): The radius of the circles to be drawn around each keypoint. Defaults to 5.

        Returns:
            Image: The PIL image object with keypoints visualized.
        """
        # Check if there are any points to draw.
        if points is not None:
            # Create a drawing context for the image.
            draw = ImageDraw.Draw(image)
            # Iterate over each point in the points list.
            for kp in points:
                # Draw an outer ellipse in blue. This represents the outer part of the keypoint marker.
                # The coordinates for the ellipse are calculated based on the keypoint position and the specified radius.
                draw.ellipse(
                    [
                        (
                            kp[0] - radius,
                            kp[1] - radius,
                        ),  # Top-left corner of the bounding box
                        (
                            kp[0] + radius,
                            kp[1] + radius,
                        ),  # Bottom-right corner of the bounding box
                    ],
                    outline=UGENT.BLUE,  # The color of the outline. UGENT.BLUE is a predefined color in this context.
                    width=2,  # The thickness of the ellipse's outline.
                )
                # Draw an inner ellipse in yellow. This creates a two-tone effect for better visibility.
                draw.ellipse(
                    [
                        (
                            kp[0] - (radius - 1),
                            kp[1] - (radius - 1),
                        ),  # Slightly inside the outer ellipse
                        (
                            kp[0] + (radius - 1),
                            kp[1] + (radius - 1),
                        ),  # to create a border effect.
                    ],
                    outline=UGENT.YELLOW,  # The color of the inner ellipse's outline.
                    width=1,  # The thickness of the inner ellipse's outline.
                )

        # Return the image with the keypoints drawn on it.
        return image

    def __process_keyboard(self, key, idx, vid, path):
        """
        Handles keyboard input for navigating frames, saving data, and toggling trial
        validity. This function allows the user to interact with the video labeling interface
        using the keyboard.

        Args:
            key (int): The ASCII value of the pressed key.
            idx (int): The current index of the video frame being viewed.
            vid (Dict[str, Any]): The video data structure containing frames, keypoints, etc.
            path (Path): The path where the video data (or modifications thereof) should be saved.

        Raises:
            BreakException: Raised to exit the video labeling loop, typically triggered by pressing the ESC key.

        Returns:
            int: The updated index after processing the keyboard input, which might have been incremented
            or decremented based on the user's navigation through video frames.
        """
        # Increment the frame index to move to the next frame if the 'd' key is pressed.
        if key == ord("d"):
            idx += 1

        # Decrement the frame index to move to the previous frame if the 'a' key is pressed.
        if key == ord("a"):
            idx -= 1

        # Save the current state of the video (including any labeled keypoints) to the specified path
        # when the 'l' key is pressed.
        if key == ord("l"):
            self.__save(vid, path)

        # Toggle the validity of the current trial when the 'v' key is pressed. This might be used
        # to mark a trial as invalid or valid based on the user's assessment.
        if key == ord("v"):
            self.__toggle_valid_trial(vid)

        # Raise a BreakException to signal that the user wants to exit the labeling interface,
        # typically by pressing the ESC key (ASCII value 27).
        if key == 27:
            raise BreakException()

        # Return the updated frame index after processing the keyboard input.
        return idx

    def __process_mouse(self, event, coords, points, idx):
        """
        Processes mouse clicks for adding or removing keypoints. When the left mouse button is
        clicked near an existing keypoint, that keypoint is removed. If the click is not near
        any keypoint, a new keypoint is added at the click location.

        Args:
            event (int): The OpenCV mouse event type.
            coords (Tuple[int, int]): The (x, y) coordinates of the mouse click.
            points (List[np.ndarray]): The list of keypoints, where each keypoint is represented
                                    as an array of its (x, y) coordinates.
            idx (int): The current index of the frame being edited.

        Returns:
            Tuple[List[np.ndarray], int]: The updated list of keypoints after processing the
                                        mouse click and the current frame index.
        """
        # Check if the left mouse button was clicked.
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert the click coordinates to a NumPy array for easier manipulation.
            coords = np.array(coords)
            # If there are no keypoints for the current frame, or it's an empty list, add the clicked point as the first keypoint.
            if points[idx] is None or len(points[idx]) == 0:
                points[idx] = coords[None, :]
            else:
                # Calculate the Euclidean distance from the click to each existing keypoint.
                D = np.linalg.norm(points[idx] - coords[None, :], axis=-1)
                # Find the closest keypoint to the click.
                argmin = np.argmin(D, axis=0)
                # If the closest keypoint is within a threshold distance, remove it.
                if D[argmin] < self.config["min_distance_between_keypoints"]:
                    points[idx] = np.concatenate(
                        (points[idx][:argmin], points[idx][(argmin + 1) :]), axis=0
                    )
                else:
                    # If no keypoint is close enough, add a new keypoint at the click location.
                    points[idx] = np.concatenate((points[idx], coords[None, :]), axis=0)

                # Invalidate keypoints for all subsequent frames to ensure they are recalculated or revalidated.
                for jj in range(idx + 1, len(points)):
                    points[jj] = None

            # Simulate pressing the 'enter' key using pyautogui to trigger cv2 window update (e.g., refresh the display).
            pyautogui.press("enter")

        # Return the updated keypoints list and the current frame index.
        return points, idx

    def __toggle_valid_trial(self, vid):
        """
        Toggles the validity state of a trial. If a trial is currently marked as valid, this function
        will mark it as invalid, and vice versa. This is useful for managing trials that may not meet
        certain criteria or need to be reviewed. (e.g. object not grasped properly)

        Args:
            vid (Dict[str, Any]): The video data structure for the current trial. It must contain a
                                'valid' key that indicates whether the trial is considered valid.
        """
        # Toggle the 'valid' state. If 'vid["valid"]' is True, it becomes False, and if it's False, it becomes True.
        vid["valid"] = not vid["valid"]

    def __save(self, vid: Dict[str, Any], path: Path):
        """
        Saves the labeled keypoints and other state information to a JSON file. This method
        serializes the video data structure (which includes keypoints and other metadata)
        into a JSON format and writes it to a file, effectively persisting the labeling work
        done on a video trial.

        Args:
            vid (Dict[str, Any]): The video data structure containing information such as keypoints,
                                validity state, and other metadata that needs to be saved.
            path (Path): The directory path where the JSON file should be saved. The file will be
                        named 'state.json' within this directory.
        """
        # Initialize an empty dictionary to store the data to be saved.
        output = {}
        # Copy relevant information from the 'vid' dictionary to the 'output' dictionary.
        output["state_sophie"] = vid["state_sophie"]
        output["rs2_intrinsics"] = vid[
            "rs2_intrinsics"
        ]  # Camera intrinsics or similar data.

        # Prepare the keypoints for JSON serialization. JSON does not support NumPy array directly,
        # so they must be converted to lists.
        output["keypoints"] = []
        for kp in vid["keypoints"]:
            if kp is None:
                output["keypoints"].append([])
            elif isinstance(kp, np.ndarray):
                # Convert NumPy arrays to lists for JSON compatibility.
                output["keypoints"].append(kp.tolist())

        # Include the validity state of the trial in the output.
        output["valid"] = vid["valid"]

        # Open the 'state.json' file in write mode and dump the 'output' dictionary into it as JSON.
        with open(path / "state.json", "w") as f:
            json.dump(output, f, indent=2)  # Use an indent for pretty-printing.

        # Log the save operation to the console or a log file, indicating where the data was saved.
        pyout(f"Saved @ {str(path)}")


def main():
    node = KeypointLabeler()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
