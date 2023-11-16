import numpy as np

CONF_GRAVITY = np.array([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
])
CONF_LINK_GRAVITY = np.array([
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, -1],
])
CONF_LINKS = np.array([1, 3, 2])
CONF_LINK_TYPES = np.array([
    [0, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
])
CONF_COLORS = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
])
