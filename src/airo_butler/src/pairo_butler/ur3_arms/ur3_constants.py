import numpy as np

# ip addresses
IP_RIGHT_UR3 = "10.42.0.162"
IP_LEFT_UR3 = "10.42.0.163"

# standard poses left arm
# POSE_LEFT_REST = np.array([0.0, -0.25, -0.5, -0.75, 0.5, 0]) * np.pi
# POSE_LEFT_PRESENT = np.array([0.5, - 0.75, 0, -0.5, 0, 0]) * np.pi
POSE_LEFT_SCAN = np.array([+0.00, -0.55, +0.00, -0.50, +0.50, +0.00]) * np.pi

# POSE_RIGHT_CLOCK = np.array([-0.70, -1.0, +0.00, +0.00, -0.30, +0.00]) * np.pi
POSE_RIGHT_CLOCK = np.array([-0.50, -1.0, +0.00, +0.00, -0.35, +0.00]) * np.pi
POSE_RIGHT_MIDDLE = np.array([+0.00, -1.0, +0.50, -0.50, -0.50, +0.00]) * np.pi
POSE_RIGHT_COUNTER = np.array([+0.60, -1.00, +0.25, -0.25, -0.75, +0.00]) * np.pi
