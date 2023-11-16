from typing import Tuple

import numpy as np

from app.constants import COL_CY, COL_CX
from app.utils import interact_all, apply_speed


class Storage:
    data: np.ndarray
    _max_coord: np.ndarray
    _cluster_size: int
    _gravity: np.ndarray
    _link_gravity: np.ndarray
    _links: np.array
    _type_links: np.array
    _colors: np.array

    def __init__(self, size: int, max_coord: Tuple[int, int], cluster_size: int):
        self._max_coord = np.array(max_coord)
        self._cluster_size = cluster_size

        self._gravity = np.array([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ])
        self._link_gravity = np.array([
            [-1, -1, 1],
            [-1, -1, -1],
            [-1, -1, -1],
        ])
        self._links = np.array([1, 3, 2])
        self._type_links = np.array([
            [0, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
        ])

        self.data = np.array([
            np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
            np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.repeat(1, size).astype('float'),
            np.repeat(0, size).astype('float'),
            np.repeat(0, size).astype('float'),
            np.random.randint(low=0, high=3, size=size).astype('float'),
        ], dtype=np.float64).T

    def interact(self) -> None:
        clusters_coords = np.unique(self.data[:, [COL_CX, COL_CY]], axis=0)
        interact_all(self.data, clusters_coords)

    def move(self) -> None:
        apply_speed(self.data, self._max_coord, self._cluster_size)
