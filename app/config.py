import torch

ATOMS_COUNT = 1000
WINDOW_SIZE = (1920, 1080)
MAX_COORD = (1000, 1000)
SIMULATION_SPEED = 1

MAX_INTERACTION_DISTANCE = 100
MIN_LINK_DISTANCE = 30
MAX_LINK_DISTANCE = 70

FORCE_BOUNCE_ELASTIC = 0.3
FORCE_NOT_LINKED_GRAVITY = 10
FORCE_LINKED_GRAVITY = 10
FORCE_LINKED_ELASTIC = 0.16

INERTIAL_FACTOR = 0.9

ATOMS_GRAVITY = torch.tensor([
    # [-1, -1, -1],
    # [-1, -1, -1],
    # [-1, -1, -1],

    # [1, -1, -0.2],
    # [-1, 0, 0.5],
    # [0.1, -0.1, 0.1],

    [0.2, -1, -0.2],
    [-1, 0, 0.5],
    [0.1, -0.3, 0.1],
], dtype=torch.float64)
ATOMS_LINK_GRAVITY = torch.tensor([
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, -1],
], dtype=torch.float64)
ATOMS_LINKS = torch.tensor([1, 3, 2], dtype=torch.int64)
ATOMS_LINK_TYPES = torch.tensor([
    [0, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
], dtype=torch.int64)
ATOMS_COLORS = torch.tensor([
    [250, 20, 20],
    [200, 140, 100],
    [80, 170, 140],
], dtype=torch.int64)
ATOMS_RADIUS = torch.tensor([4, 5, 3], dtype=torch.int64)
