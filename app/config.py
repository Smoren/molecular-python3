import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 1000
WINDOW_SIZE = (1920, 1080)
MAX_COORD = (1920, 1080)
SIMULATION_SPEED = 1

MAX_INTERACTION_DISTANCE = 100
FORCE_GRAVITY = 1
INERTIAL_FACTOR = 0.9

ATOMS_LJ_PARAMS = np.array([
    [5, 0.1],
    [4, 0.2],
    [3, 0.3],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([5, 5, 5], dtype=np.int64)
