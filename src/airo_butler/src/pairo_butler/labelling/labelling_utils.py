import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pairo_butler.utils.tools import load_mp4_video, pyout
import rospy as ros


class LabellingUtils:
    @staticmethod
    def load_trial_data(path: Path, include_video: bool = True):
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
        try:
            with open(path / "state.json", "r") as f:
                state: Dict[str, Any] = json.load(f)

            # Load the video frames from the 'video.mp4' file located in the same directory
            if include_video:
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

    @staticmethod
    def save_trial_data(path: Path, data: Dict[str, Any]) -> None:
        if "frames" in data:
            del data["frames"]

        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = val.tolist()

        with open(path / "state.json", "w") as f:
            json.dump(data, f, indent=2)
