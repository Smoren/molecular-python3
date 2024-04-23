from typing import Tuple

import numpy as np
import torch

from app.config import ATOMS_COLORS, ATOMS_RADIUS


def generate_atoms(size: int, max_coord: Tuple[int, int]):
    types = np.random.randint(low=0, high=len(ATOMS_COLORS), size=size)
    radius = ATOMS_RADIUS[types]

    return torch.tensor([
        # ID
        np.arange(0, size),
        # Coords
        np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
        np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
        # Speed
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Radius
        radius.to(torch.float64),
        # Cluster
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        # Type
        types.astype('float'),
        # Links counter
        np.repeat(0, size).astype('float'),
        # Links type counters
        *(np.repeat(0, size).astype('float') for _ in range(len(ATOMS_COLORS)))
    ], dtype=torch.float64).T


def generate_debug():
    return torch.tensor([
        # ID
        np.array([0, 1, 2, 3]).astype('float'),
        # Coords
        np.array([580, 580, 600, 600]).astype('float'),
        np.array([580, 600, 580, 600]).astype('float'),
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
    ], dtype=torch.float64).T


def generate_debug2():
    return torch.tensor([
        # ID
        np.array([0, 1]).astype('float'),
        # Coords
        np.array([550, 550]).astype('float'),
        np.array([550, 600]).astype('float'),
        # Speed
        np.array([0, 0]).astype('float'),
        np.array([0, 0]).astype('float'),
        # Radius
        np.array([5, 10]).astype('float'),
        # Cluster
        np.array([0, 0]).astype('float'),
        np.array([0, 0]).astype('float'),
        # Type
        np.array([1, 2]).astype('float'),
        # Links counter
        np.array([0, 0]).astype('float'),
        # Links type counters
        np.array([0, 0]).astype('float'),
        np.array([0, 0]).astype('float'),
        np.array([0, 0]).astype('float'),
    ], dtype=torch.float64).T
