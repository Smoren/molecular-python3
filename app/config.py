import numpy as np

MODE_DEBUG = False

ATOMS_COUNT = 10000
WINDOW_SIZE = (1920, 1080)
MAX_COORD = (2500, 2500)
SIMULATION_SPEED = 1

MAX_INTERACTION_DISTANCE = 100
MIN_LINK_DISTANCE = 30
MAX_LINK_DISTANCE = 70

FORCE_BOUNCE_ELASTIC = 0.3
FORCE_NOT_LINKED_GRAVITY = 10
FORCE_LINKED_GRAVITY = 10
FORCE_LINKED_ELASTIC = 0.16

INERTIAL_FACTOR = 0.9

ATOMS_GRAVITY = np.array([
    # [-1, -1, -1],
    # [-1, -1, -1],
    # [-1, -1, -1],

    # [1, -1, -0.2],
    # [-1, 0, 0.5],
    # [0.1, -0.1, 0.1],

    [0.2, -1, -0.2],
    [-1, 0, 0.5],
    [0.1, -0.3, 0.1],
], dtype=np.float64)
ATOMS_LINK_GRAVITY = np.array([
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, -1],
], dtype=np.float64)
ATOMS_LINKS = np.array([1, 3, 2], dtype=np.int64)
ATOMS_LINK_TYPES = np.array([
    [0, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
], dtype=np.int64)
ATOMS_COLORS = np.array([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=np.int64)
ATOMS_RADIUS = np.array([4, 5, 3], dtype=np.int64)
