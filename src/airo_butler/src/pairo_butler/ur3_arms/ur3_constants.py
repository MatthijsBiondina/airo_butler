import numpy as np

# ip addresses
IP_RIGHT_UR3 = "10.42.0.162"
IP_LEFT_UR3 = "10.42.0.163"

# standard poses left arm
POSE_LEFT_REST = np.array([0., -.25, -0.5, -.75, .5, 0]) * np.pi
POSE_LEFT_PRESENT = np.array([0.5, - 0.75, 0, -0.5, 0, 0]) * np.pi

POSE_RIGHT_SCAN1 = np.array(
    [+0.100, -0.721, +0.670, -0.449, -1.000, +0.000]) * np.pi
