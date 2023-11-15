from typing import Tuple

import numpy as np
import multiprocessing as mp

from app.shared import create_shared_variable_for_cluster, destroy_shared_variable, get_shared_variable


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

    @staticmethod
    def interact_step_old(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, cluster_mask):
        for atom in cluster_atoms:
            d = np.array([
                neighbour_atoms[:, AtomField.X] - atom[AtomField.X],
                neighbour_atoms[:, AtomField.Y] - atom[AtomField.Y]]
            ).T

            l = np.linalg.norm(d, axis=1)

            du = (d.T / l).T
            du[np.isnan(du)] = 0

            dv = (du.T / l).T
            dv[np.isnan(dv)] = 0
            dv = np.sum(dv, axis=0) * 4

            atom[AtomField.VX] += dv[0]
            atom[AtomField.VY] += dv[1]

        return cluster_atoms, cluster_mask

    @staticmethod
    def interact_step(
        cluster_coords: Tuple[int, ...],
        cluster_atoms_shape: Tuple[int, ...],
        neighbour_atoms_shape: Tuple[int, ...],
    ):
        cluster_atoms, shm1 = get_shared_variable(cluster_coords, cluster_atoms_shape, 'cluster_atoms')
        neighbour_atoms, shm2 = get_shared_variable(cluster_coords, neighbour_atoms_shape, 'neighbour_atoms')

        for atom in cluster_atoms:
            d = np.array([
                neighbour_atoms[:, AtomField.X] - atom[AtomField.X],
                neighbour_atoms[:, AtomField.Y] - atom[AtomField.Y]]
            ).T

            l = np.linalg.norm(d, axis=1)

            du = (d.T / l).T
            du[np.isnan(du)] = 0

            dv = (du.T / l).T
            dv[np.isnan(dv)] = 0
            dv = np.sum(dv, axis=0) * 4

            atom[AtomField.VX] += dv[0]
            atom[AtomField.VY] += dv[1]

        return cluster_coords, shm1, shm2

    def interact(self) -> None:
        clusters_coords = np.unique(self.data[:, [AtomField.CLUSTER_X, AtomField.CLUSTER_Y]], axis=0)

        tasks_data = []
        shared_variables = []
        cluster_mask_map = dict()
        cluster_atoms_shape_map = dict()

        for cluster_coords in clusters_coords:
            cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]
            cluster_coords_tuple = (cluster_x, cluster_y)

            cluster_mask = (self.data[:, AtomField.CLUSTER_X] == cluster_x) & \
                           (self.data[:, AtomField.CLUSTER_Y] == cluster_y)
            neighbours_mask = (self.data[:, AtomField.CLUSTER_X] >= cluster_x - 1) & \
                              (self.data[:, AtomField.CLUSTER_X] <= cluster_x + 1) & \
                              (self.data[:, AtomField.CLUSTER_Y] >= cluster_y - 1) & \
                              (self.data[:, AtomField.CLUSTER_Y] <= cluster_y + 1)
            cluster_atoms = self.data[cluster_mask]
            neighbour_atoms = self.data[neighbours_mask]

            cluster_mask_map[cluster_coords_tuple] = cluster_mask
            cluster_atoms_shape_map[cluster_coords_tuple] = cluster_atoms.shape

            shared_variables.append(
                create_shared_variable_for_cluster(cluster_coords_tuple, cluster_atoms, 'cluster_atoms')
            )
            shared_variables.append(
                create_shared_variable_for_cluster(cluster_coords_tuple, neighbour_atoms, 'neighbour_atoms')
            )

            tasks_data.append((cluster_coords_tuple, cluster_atoms.shape, neighbour_atoms.shape))

        task_results = self._pool.starmap(self.interact_step, tasks_data)
        for task_result in task_results:
            cluster_coords_tuple, shm1, shm2 = task_result
            cluster_atoms, shm = get_shared_variable(
                cluster_coords_tuple,
                cluster_atoms_shape_map[cluster_coords_tuple],
                'cluster_atoms',
            )
            self.data[cluster_mask_map[cluster_coords_tuple]] = cluster_atoms

        for var in shared_variables:
            destroy_shared_variable(var)

    def move(self) -> None:
        self.data[:, AtomField.X] += self.data[:, AtomField.VX]
        self.data[:, AtomField.Y] += self.data[:, AtomField.VY]

        self.data[self.data[:, AtomField.X] < 0, AtomField.VX] += 10
        self.data[self.data[:, AtomField.Y] < 0, AtomField.VY] += 10

        self.data[self.data[:, AtomField.X] > self._max_coord[0], AtomField.VX] -= 10
        self.data[self.data[:, AtomField.Y] > self._max_coord[1], AtomField.VY] -= 10

        self.data[:, AtomField.CLUSTER_X] = np.floor(self.data[:, AtomField.X] / self._cluster_size)
        self.data[:, AtomField.CLUSTER_Y] = np.floor(self.data[:, AtomField.Y] / self._cluster_size)
