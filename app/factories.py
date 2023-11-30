from typing import Tuple

import numpy as np

from app.config import ATOMS_COLORS, ATOMS_RADIUS


def generate_atoms(size: int, max_coord: Tuple[int, int]):
    types = np.random.randint(low=0, high=len(ATOMS_COLORS), size=size)
    radius = ATOMS_RADIUS[types]

    return np.array([
        # ID
        np.arange(0, size),
        # Coords
        np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
        np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
        # Next Coords
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Speed
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Next Speed
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Force
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Next Force
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Radius
        radius.astype('float'),
        # Cluster
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Type
        types.astype('float'),
    ], dtype=np.float64).T


def generate_debug():
    return np.array([
        # ID
        np.array([0, 1, 2, 3]).astype('float'),
        # Coords
        np.array([500, 500, 600, 600]).astype('float'),
        np.array([500, 600, 500, 600]).astype('float'),
        # Next Coords
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Speed
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Next Speed
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Force
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Next Force
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Radius
        np.array([5, 5, 5, 5]).astype('float'),
        # Cluster
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Type
        np.array([1, 1, 1, 1]).astype('float'),
    ], dtype=np.float64).T
