import numpy as np

USE_JIT_CACHE = True

ATOMS_COUNT = 10000
WINDOW_SIZE = (1920, 1080)
MAX_COORD = (2500, 2500)
SIMULATION_SPEED = 1

MAX_INTERACTION_DISTANCE = 100
FORCE_GRAVITY = 1
INERTIAL_FACTOR = 0.7
MORSE_MULT = 1

ATOMS_MORSE_PARAMS = np.array([
    [0.008, 0.035, 50],
    [0.01, 0.03, 50],
    [0.1, 0.15, 10],

    # [0.06334, 0.58993, 5.336],
    # [0.05424, 0.49767, 6.369],
    # [0.4174, 1.3885, 2.845],
], dtype=np.float64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([2, 4, 3], dtype=np.int64)
