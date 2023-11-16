from typing import Tuple

import numpy as np
import multiprocessing as mp

from app.utils import interact_all, apply_speed


class Storage:
    data: np.ndarray
    _max_coord: np.ndarray
    _cluster_size: int
    _pool: mp.Pool

    def __init__(self, size: int, max_coord: Tuple[int, int], cluster_size: int):
        self._max_coord = np.array(max_coord)
        self._cluster_size = cluster_size
        self._pool = mp.Pool(processes=20)
        self.data = np.array([
            np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
            np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.repeat(1, size),
            np.repeat(0, size),
            np.repeat(0, size),
        ], dtype=np.float64).T

    def interact(self) -> None:
        _cx, _cy = 5, 6
        coords_columns = np.array([_cx, _cy])
        clusters_coords = np.unique(self.data[:, coords_columns], axis=0)
        interact_all(self.data, clusters_coords)


    def move(self) -> None:
        apply_speed(self.data, self._max_coord, self._cluster_size)
        # self.data[:, COL_X] += self.data[:, COL_VX]
        # self.data[:, COL_Y] += self.data[:, COL_VY]
        #
        # self.data[self.data[:, COL_X] < 0, COL_VX] += 10
        # self.data[self.data[:, COL_Y] < 0, COL_VY] += 10
        #
        # self.data[self.data[:, COL_X] > self._max_coord[0], COL_VX] -= 10
        # self.data[self.data[:, COL_Y] > self._max_coord[1], COL_VY] -= 10
        #
        # self.data[:, COL_CX] = np.floor(self.data[:, COL_X] / self._cluster_size)
        # self.data[:, COL_CY] = np.floor(self.data[:, COL_Y] / self._cluster_size)
