from typing import Tuple

import numpy as np

from app.config import ATOMS_COLORS


def generate_atoms(size: int, max_coord: Tuple[int, int]):
    return np.array([
        np.arange(0, size),
        np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
        np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.repeat(3, size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.random.randint(low=0, high=len(ATOMS_COLORS), size=size).astype('float'),
    ], dtype=np.float64).T
