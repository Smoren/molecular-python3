from typing import Tuple

import numpy as np

from app.config import CONF_COLORS


def generate_atoms(size: int, max_coord: Tuple[int, int]):
    return np.array([
        np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
        np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
        np.random.randint(low=-10, high=10, size=size).astype('float'),
        np.random.randint(low=-10, high=10, size=size).astype('float'),
        np.repeat(1, size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.repeat(0, size).astype('float'),
        np.random.randint(low=0, high=len(CONF_COLORS), size=size).astype('float'),
    ], dtype=np.float64).T
