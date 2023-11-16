from typing import List, Tuple

import numpy as np

from numba import jit


@jit(fastmath=True, parallel=True, nopython=True)
def clusterize_tasks(data: np.ndarray):
    _cx, _cy = 5, 6
    tasks_data = []
    clusters_coords = np.unique(data[:, [_cx, _cy]], axis=0)

    for cluster_coords in clusters_coords:
        cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

        cluster_mask = (data[:, _cx] == cluster_x) & \
                       (data[:, _cy] == cluster_y)
        neighbours_mask = (data[:, _cx] >= cluster_x - 1) & \
                          (data[:, _cx] <= cluster_x + 1) & \
                          (data[:, _cy] >= cluster_y - 1) & \
                          (data[:, _cy] <= cluster_y + 1)
        cluster_atoms = data[cluster_mask]
        neighbour_atoms = data[neighbours_mask]
        tasks_data.append((cluster_atoms, neighbour_atoms, cluster_mask))

    return tasks_data
