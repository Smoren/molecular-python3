from typing import Tuple

import numpy as np

from app.config import ATOMS_COLORS


def generate_atoms(size: int, max_coord: Tuple[int, int]):
    return np.array([
        # ID
        np.arange(0, size),
        # Coords
        np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
        np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
        # Speed
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Radius
        np.repeat(3, size).astype('float'),
        # Cluster
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Type
        np.random.randint(low=0, high=len(ATOMS_COLORS), size=size).astype('float'),
        # Links counter
        np.repeat(0, size).astype('float'),
        # Links type counters
        *(np.repeat(0, size).astype('float') for _ in range(len(ATOMS_COLORS)))
    ], dtype=np.float64).T


def generate_debug():
    return np.array([
        # ID
        np.array([0, 1, 2, 3]).astype('float'),
        # Coords
        np.array([500, 500, 600, 600]).astype('float'),
        np.array([500, 600, 500, 600]).astype('float'),
        # Speed
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Radius
        np.array([5, 5, 5, 5]).astype('float'),
        # Cluster
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        # Type
        np.array([1, 1, 1, 1]).astype('float'),
        # Links counter
        np.array([0, 0, 0, 0]).astype('float'),
        # Links type counters
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
        np.array([0, 0, 0, 0]).astype('float'),
    ], dtype=np.float64).T
