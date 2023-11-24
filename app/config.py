import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 15000
WINDOW_SIZE = (1920, 1080)
MAX_COORD = (3000, 3000)
SIMULATION_SPEED = 1

MAX_INTERACTION_DISTANCE = 100
FORCE_GRAVITY = 1
INERTIAL_FACTOR = 0.8
MORSE_MULT = 1

ATOMS_MORSE_PARAMS = np.array([
    [[0.01, 0.035, 50], [0.008, 0.035, 70], [0.008, 0.035, 70], [0.001, 0.01, 100]],
    [[0.01, 0.03, 50], [0.01, 0.03, 50], [0.01, 0.03, 50], [0.001, 0.01, 200]],
    [[0.02, 0.15, 30], [0.1, 0.15, 15], [0.1, 0.1, 15], [0.001, 0.01, 100]],
    [[1, 1, 3], [0.1, 0.05, 50], [0.1, 0.01, 100], [0.01, 0.01, 100]],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
    [250, 0, 250],
], dtype=np.int64)
ATOMS_RADIUS = np.array([3, 5, 4, 2], dtype=np.int64)
