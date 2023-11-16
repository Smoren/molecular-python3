import numpy as np
import numba as nb

from app.constants import COL_CX, COL_CY, COL_X, COL_Y, COL_VX, COL_VY


@nb.jit(
    (nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64[:])),
    fastmath=True,
    nopython=True,
    cache=True,
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
    cache=True,
    looplift=True,
    boundscheck=False,
)
def clusterize_tasks(data: np.ndarray, clusters_coords: np.ndarray) -> list:
    return [get_cluster_task_data(data, clusters_coords[i]) for i in nb.prange(clusters_coords.shape[0])]


@nb.jit(
    (nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64[:, :], nb.boolean[:])),
    fastmath=True,
    nopython=True,
    cache=True,
    looplift=True,
    boundscheck=False,
)
def interact_cluster(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, cluster_mask: np.ndarray):
        coords_columns = np.array([COL_X, COL_Y])

        for i in nb.prange(cluster_atoms.shape[0]):
            atom = cluster_atoms[i]
            mask = (neighbour_atoms[:, COL_X] != atom[COL_X]) | (neighbour_atoms[:, COL_Y] != atom[COL_Y])

            d = neighbour_atoms[mask][:, coords_columns] - atom[coords_columns]
            l = np.sqrt(d[:, 0]**2 + d[:, 1]**2)

            du = (d.T / l).T
            dv = (du.T / l).T
            dv = np.sum(dv, axis=0) * 4

            atom[COL_VX] += dv[0]
            atom[COL_VY] += dv[1]

        return cluster_atoms, cluster_mask


@nb.jit(
    (nb.types.NoneType('none')(nb.float64[:, :], nb.float64[:, :])),
    fastmath=True,
    nopython=True,
    cache=True,
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
    (nb.types.NoneType('none')(nb.float64[:, :], nb.int64[:], nb.int64)),
    fastmath=True,
    nopython=True,
    cache=True,
    looplift=True,
    boundscheck=False,
)
def apply_speed(data: np.ndarray, max_coord: np.ndarray, cluster_size: int) -> None:
    data[:, COL_X] += data[:, COL_VX]
    data[:, COL_Y] += data[:, COL_VY]

    data[data[:, COL_X] < 0, COL_VX] += 10
    data[data[:, COL_Y] < 0, COL_VY] += 10

    data[data[:, COL_X] > max_coord[0], COL_VX] -= 10
    data[data[:, COL_Y] > max_coord[1], COL_VY] -= 10

    data[:, COL_CX] = np.floor(data[:, COL_X] / cluster_size)
    data[:, COL_CY] = np.floor(data[:, COL_Y] / cluster_size)
