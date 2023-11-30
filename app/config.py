import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 20000
WINDOW_SIZE = (500, 500)
MAX_COORD = (500, 500)
MAX_INTERACTION_DISTANCE = 30
DELTA_T = 0.01

ATOMS_LJ_PARAMS = np.array([
    [1.1, 0.6],
    [1.0, 0.7],
    [0.9, 0.8],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([1, 1, 1], dtype=np.int64)
