from pathlib import Path
from pairo_butler.utils.tools import pyout


class Preprocessor:
    def __init__(self):
        pass


if __name__ == "__main__":
    root_in = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS_raw/"
    )
    root_ou = Path(
        "/home/matt/catkin_ws/src/airo_butler/src/pairo_butler/SplaTAM/data/TOWELS/"
    )

    pyout()
