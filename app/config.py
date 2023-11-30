import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 10000
WINDOW_SIZE = (1500, 1000)
MAX_COORD = (500, 500)
MAX_INTERACTION_DISTANCE = 30
DELTA_T = 0.1
GRAVITY = 0.2
INERTIA = 0.999

ATOMS_LJ_PARAMS = np.array([
    [1.0, 2.0, 2.0],
    [1.0, 0.3, 5.0],
    [1.1, 0.3, 4.0],
    [1.2, 0.3, 3.0],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [255, 255, 255],
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([1, 1, 1, 1], dtype=np.int64)
