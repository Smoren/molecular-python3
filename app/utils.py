import numpy as np
import numba as nb

from app.constants import COL_CX, COL_CY, COL_X, COL_Y, COL_VX, COL_VY, COL_TYPE, COL_R
from app.config import ATOMS_GRAVITY, CLUSTER_SIZE


@nb.jit(
    (nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64[:])),
    fastmath=True,
    nopython=True,
    # cache=True,
    boundscheck=False,
)
def get_cluster_task_data(data: np.ndarray, cluster_coords: np.ndarray) -> tuple:
    cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

    cluster_mask = (data[:, COL_CX] == cluster_x) & \
                   (data[:, COL_CY] == cluster_y)
    neighbours_mask = (data[:, COL_CX] >= cluster_x - 1) & \
                      (data[:, COL_CX] <= cluster_x + 1) & \
                      (data[:, COL_CY] >= cluster_y - 1) & \
                      (data[:, COL_CY] <= cluster_y + 1)

    return data[cluster_mask], data[neighbours_mask], cluster_mask


@nb.jit(
    (nb.types.List(nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:])))(nb.float64[:, :], nb.float64[:, :])),
    fastmath=True,
    nopython=True,
    # cache=True,
    looplift=True,
    boundscheck=False,
)
def clusterize_tasks(data: np.ndarray, clusters_coords: np.ndarray) -> list:
    return [get_cluster_task_data(data, clusters_coords[i]) for i in nb.prange(clusters_coords.shape[0])]


@nb.jit(
    (nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64[:, :], nb.boolean[:])),
    fastmath=True,
    nopython=True,
    # cache=True,
    looplift=True,
    boundscheck=False,
)
def interact_cluster(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, cluster_mask: np.ndarray):
        coords_columns = np.array([COL_X, COL_Y])

        for i in nb.prange(cluster_atoms.shape[0]):
            atom = cluster_atoms[i]
            mask_exclude_self = (neighbour_atoms[:, COL_X] != atom[COL_X]) | (neighbour_atoms[:, COL_Y] != atom[COL_Y])

            neighbours = neighbour_atoms[mask_exclude_self]
            d = neighbours[:, coords_columns] - atom[coords_columns]
            l2 = d[:, 0]**2 + d[:, 1]**2
            l = np.sqrt(l2)

            mask_in_radius = l <= CLUSTER_SIZE
            neighbours = neighbours[mask_in_radius]
            d = d[mask_in_radius]
            l = l[mask_in_radius]

            mask_bounced = l < neighbours[:, COL_R] + atom[COL_R]
            neighbours_nb = neighbours[~mask_bounced]
            nb_d = d[~mask_bounced]
            nb_l = l[~mask_bounced]

            mult = ATOMS_GRAVITY[int(atom[COL_TYPE]), neighbours_nb[:, COL_TYPE].astype(np.int64)]

            nb_nd = (nb_d.T / nb_l).T
            nb_dv = (nb_nd.T / nb_l).T  # l2 вместо l ???
            nb_dv = (nb_dv.T * mult).T
            nb_dv = np.sum(nb_dv, axis=0) * 3  # TODO factor

            b_d = d[mask_bounced]
            b_l = l[mask_bounced]

            b_nd = (b_d.T / b_l).T
            b_dv = (b_nd.T / np.maximum(b_l, 2)).T  # TODO factor
            b_dv = np.sum(b_dv, axis=0) * 0.5  # TODO factor

            atom[COL_VX] += nb_dv[0] - b_dv[0]
            atom[COL_VY] += nb_dv[1] - b_dv[1]

        return cluster_atoms, cluster_mask


@nb.jit(
    (nb.types.NoneType('none')(nb.float64[:, :], nb.float64[:, :])),
    fastmath=True,
    nopython=True,
    # cache=True,
    looplift=True,
    boundscheck=False,
    parallel=True,
)
def interact_all(data: np.ndarray, clusters_coords: np.ndarray) -> None:
    tasks = clusterize_tasks(data, clusters_coords)
    for i in nb.prange(len(tasks)):
        task_data = tasks[i]
        cluster_atoms, cluster_mask = interact_cluster(*task_data)
        data[cluster_mask] = cluster_atoms


@nb.jit(
    (nb.types.NoneType('none')(nb.float64[:, :], nb.int64[:])),
    fastmath=True,
    nopython=True,
    # cache=True,
    looplift=True,
    boundscheck=False,
)
def apply_speed(data: np.ndarray, max_coord: np.ndarray) -> None:
    data[:, COL_X] += data[:, COL_VX]
    data[:, COL_Y] += data[:, COL_VY]

    mask_x_min = data[:, COL_X] < 0
    mask_y_min = data[:, COL_Y] < 0
    mask_x_max = data[:, COL_X] > max_coord[0]
    mask_y_max = data[:, COL_Y] > max_coord[1]

    data[mask_x_min, COL_VX] *= -1
    data[mask_y_min, COL_VY] *= -1
    data[mask_x_min, COL_X] *= -1
    data[mask_y_min, COL_Y] *= -1

    data[mask_x_max, COL_VX] *= -1
    data[mask_y_max, COL_VY] *= -1
    data[mask_x_max, COL_X] = max_coord[0] - (data[mask_x_max, COL_X] - max_coord[0])
    data[mask_y_max, COL_Y] = max_coord[1] - (data[mask_y_max, COL_Y] - max_coord[1])

    data[:, COL_VX] *= 0.98  # TODO factor
    data[:, COL_VY] *= 0.98  # TODO factor

    data[:, COL_CX] = np.floor(data[:, COL_X] / CLUSTER_SIZE)
    data[:, COL_CY] = np.floor(data[:, COL_Y] / CLUSTER_SIZE)
