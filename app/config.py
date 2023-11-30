import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 3000
WINDOW_SIZE = (1000, 1000)
MAX_COORD = (1000, 1000)
MAX_INTERACTION_DISTANCE = 50
DELTA_T = 0.1
GRAVITY = 0
INERTIA = 0.99

# ATOMS_LJ_PARAMS = np.array([
#     [1.0, 2.0, 2.0],
#     [1.0, 0.3, 5.0],
#     [1.1, 0.3, 4.0],
#     [1.2, 0.3, 3.0],
#     [0.5, 0.8, 3.5],
# ], dtype=np.float64)
# ATOMS_COLORS = np.array([
#     [255, 255, 255],
#     [250, 20, 20],
#     [200, 140, 100],
#     [80, 170, 140],
#     [170, 80, 140],
# ], dtype=np.int64)
# ATOMS_RADIUS = np.array([1, 1, 1, 1, 1], dtype=np.int64)
ATOMS_LJ_PARAMS = np.array([
    [[2.0, 0.3, 5.0], [0.8, 0.3, 2.0], [0.6, 0.3, 1.0]],
    [[0.2, 0.6, 4.0], [1.1, 0.4, 2.0], [1.1, 0.7, 4.0]],
    [[1.2, 0.3, 7.5], [1.2, 0.3, 3.0], [1.2, 0.3, 2.5]],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([1, 1, 1], dtype=np.int64)
