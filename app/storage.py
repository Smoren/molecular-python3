from typing import Tuple

import numpy as np
import multiprocessing as mp

from numba import jit


class AtomField:
    X = 0
    Y = 1
    VX = 2
    VY = 3
    RADIUS = 4
    CLUSTER_X = 5
    CLUSTER_Y = 6


class Storage:
    data: np.ndarray
    _max_coord: Tuple[int, int]
    _cluster_size: int
    _pool: mp.Pool

    def __init__(self, size: int, max_coord: Tuple[int, int], cluster_size: int):
        self._max_coord = max_coord
        self._cluster_size = cluster_size
        self._pool = mp.Pool(processes=20)
        self.data = np.array([
            np.random.randint(low=0, high=max_coord[0], size=size).astype('float'),
            np.random.randint(low=0, high=max_coord[1], size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.random.randint(low=-10, high=10, size=size).astype('float'),
            np.repeat(5, size),
            np.repeat(0, size),
            np.repeat(0, size),
        ], dtype=np.float64).T

    def interact(self) -> None:
        clusters_coords = np.unique(self.data[:, [AtomField.CLUSTER_X, AtomField.CLUSTER_Y]], axis=0)

        for cluster_coords in clusters_coords:
            cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

            cluster_mask = (self.data[:, AtomField.CLUSTER_X] == cluster_x) & \
                           (self.data[:, AtomField.CLUSTER_Y] == cluster_y)
            neighbours_mask = (self.data[:, AtomField.CLUSTER_X] >= cluster_x - 1) & \
                              (self.data[:, AtomField.CLUSTER_X] <= cluster_x + 1) & \
                              (self.data[:, AtomField.CLUSTER_Y] >= cluster_y - 1) & \
                              (self.data[:, AtomField.CLUSTER_Y] <= cluster_y + 1)
            cluster_atoms = self.data[cluster_mask]
            neighbour_atoms = self.data[neighbours_mask]

            self.data[cluster_mask], _ = self._interact_cluster(cluster_atoms, neighbour_atoms, cluster_mask)

    def move(self) -> None:
        self.data[:, AtomField.X] += self.data[:, AtomField.VX]
        self.data[:, AtomField.Y] += self.data[:, AtomField.VY]

        self.data[self.data[:, AtomField.X] < 0, AtomField.VX] += 10
        self.data[self.data[:, AtomField.Y] < 0, AtomField.VY] += 10

        self.data[self.data[:, AtomField.X] > self._max_coord[0], AtomField.VX] -= 10
        self.data[self.data[:, AtomField.Y] > self._max_coord[1], AtomField.VY] -= 10

        self.data[:, AtomField.CLUSTER_X] = np.floor(self.data[:, AtomField.X] / self._cluster_size)
        self.data[:, AtomField.CLUSTER_Y] = np.floor(self.data[:, AtomField.Y] / self._cluster_size)

    @staticmethod
    def _interact_cluster(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, cluster_mask):
        _x, _y, _vx, _vy = 0, 1, 2, 3

        for atom in cluster_atoms:
            mask = (neighbour_atoms[:, _x] != atom[_x]) | (neighbour_atoms[:, _y] != atom[_y])
            d = neighbour_atoms[mask][:, [_x, _y]] - atom[[_x, _y]]
            l = np.linalg.norm(d, axis=1)

            du = (d.T / l).T

            dv = (du.T / l).T
            dv = np.sum(dv, axis=0) * 4

            atom[_vx] += dv[_x]
            atom[_vy] += dv[_y]

        return cluster_atoms, cluster_mask
