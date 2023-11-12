from typing import Tuple

import numpy as np


class AtomField:
    X = 0
    Y = 1
    VX = 2
    VY = 3
    RADIUS = 4


class Storage:
    data: np.ndarray
    _max_coord: Tuple[int, int]

    def __init__(self, size: int, max_coord: Tuple[int, int]):
        self._max_coord = max_coord
        self.data = np.array([
            np.random.randint(low=0, high=max_coord[0], size=size),
            np.random.randint(low=0, high=max_coord[1], size=size),
            np.random.randint(low=-10, high=10, size=size),
            np.random.randint(low=-10, high=10, size=size),
            np.repeat(1, size),
        ]).T

    def move(self) -> None:
        self.data[:, AtomField.X] += self.data[:, AtomField.VX]
        self.data[:, AtomField.Y] += self.data[:, AtomField.VY]

        self.data[self.data[:, AtomField.X] < 0, AtomField.VX] += 10
        self.data[self.data[:, AtomField.Y] < 0, AtomField.VY] += 10

        self.data[self.data[:, AtomField.X] > self._max_coord[0], AtomField.VX] -= 10
        self.data[self.data[:, AtomField.Y] > self._max_coord[1], AtomField.VY] -= 10
