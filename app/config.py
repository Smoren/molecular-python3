import numpy as np

MODE_DEBUG = True

# ATOMS_COUNT = 4
# WINDOW_SIZE = (100, 100)

ATOMS_COUNT = 500
WINDOW_SIZE = (1900, 1000)

CLUSTER_SIZE = 100

ATOMS_GRAVITY = np.array([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    # [1, -1, -0.2],
    # [-1, 0, 0.5],
    # [0.1, -0.1, 0.1],
])
ATOMS_LINK_GRAVITY = np.array([
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, -1],
])
ATOMS_LINKS = np.array([1, 3, 2])
ATOMS_LINK_TYPES = np.array([
    [0, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
])
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
])
