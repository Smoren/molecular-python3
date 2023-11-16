import numpy as np
import numba as nb


@nb.jit(
    (nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64[:])),
    fastmath=True,
    nopython=True,
    cache=True,
)
def get_task_data(data: np.ndarray, cluster_coords: np.ndarray) -> tuple:
    _cx, _cy = 5, 6
    cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

    cluster_mask = (data[:, _cx] == cluster_x) & \
                   (data[:, _cy] == cluster_y)
    neighbours_mask = (data[:, _cx] >= cluster_x - 1) & \
                      (data[:, _cx] <= cluster_x + 1) & \
                      (data[:, _cy] >= cluster_y - 1) & \
                      (data[:, _cy] <= cluster_y + 1)

    return data[cluster_mask], data[neighbours_mask], cluster_mask


@nb.jit(
    (nb.types.List(nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:])))(nb.float64[:, :], nb.float64[:, :])),
    fastmath=True,
    nopython=True,
    cache=True,
)
def clusterize_tasks(data: np.ndarray, clusters_coords: np.ndarray) -> list:
    return [get_task_data(data, cluster_coords) for cluster_coords in clusters_coords]


@nb.jit(
    (nb.types.Tuple((nb.float64, nb.float64))(nb.float64[:, :], nb.float64[:])),
    fastmath=True,
    nopython=True,
    cache=True,
)
def handle_delta_speed(d: np.ndarray, l: np.ndarray):
    _x, _y, _vx, _vy = 0, 1, 2, 3

    du = (d.T / l).T

    dv = (du.T / l).T
    dv = np.sum(dv, axis=0) * 4

    return dv[_x], dv[_y]
