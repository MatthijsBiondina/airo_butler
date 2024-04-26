from pathlib import Path
import numpy as np
import rospkg


class TransformationMatrixRS2Sophie:
    """
    A singleton class that encapsulates the homogeneous transformation matrix from the 
    wrist-mounted RealSense camera to the Sophie TCP.

    This class ensures that the transformation matrix is loaded from a file only once and 
    reused, conserving resources and ensuring consistency.

    Attributes:
        __matrix (np.ndarray): The 4x4 transformation matrix loaded from a NumPy file.
    """

    _instance = None  # Singleton instance holder

    def __new__(cls, *args, **kwargs):
        """Override of the __new__ method to ensure only one instance of the class exists."""
        if not cls._instance:
            cls._instance = super(TransformationMatrixRS2Sophie, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def __init__(self):
        """Initializes the transformation matrix by loading it from a specified path."""
        # Define the path to the numpy file that contains the transformation matrix
        path: Path = (
            Path(rospkg.RosPack().get_path("airo_butler"))
            / "res"
            / "camera_tcps"
            / "T_rs2_tcp_sophie.npy"
        )
        # Load the transformation matrix from the specified file
        self.__matrix: np.ndarray = np.load(path)

    @property
    def M(self):
        """Provide read-only access to the transformation matrix."""
        return self.__matrix
