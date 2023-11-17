import numpy as np
import numba as nb

from app.constants import A_COL_CX, A_COL_CY, A_COL_X, A_COL_Y, A_COL_VX, A_COL_VY, A_COL_TYPE, A_COL_R, A_COL_ID, \
    L_COL_LHS, L_COL_RHS
from app.config import ATOMS_GRAVITY, CLUSTER_SIZE, MODE_DEBUG


@nb.njit(
    fastmath=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def isin(a, b):
    out = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        out[i] = a[i] in b
    return out


@nb.jit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:])
    ),
    fastmath=True,
    nopython=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def get_cluster_task_data(data: np.ndarray, links: np.ndarray, cluster_coords: np.ndarray) -> tuple:
    cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

    cluster_mask = (data[:, A_COL_CX] == cluster_x) & \
                   (data[:, A_COL_CY] == cluster_y)
    neighbours_mask = (data[:, A_COL_CX] >= cluster_x - 1) & \
                      (data[:, A_COL_CX] <= cluster_x + 1) & \
                      (data[:, A_COL_CY] >= cluster_y - 1) & \
                      (data[:, A_COL_CY] <= cluster_y + 1)

    cluster_atoms, neighbours_atoms = data[cluster_mask], data[neighbours_mask]

    mask_links = (isin(links[:, L_COL_LHS], cluster_atoms[:, A_COL_ID])
                  | isin(links[:, L_COL_RHS], cluster_atoms[:, A_COL_ID]))
    links_filtered = links[mask_links]

    return cluster_atoms, neighbours_atoms, links_filtered, cluster_mask


@nb.jit(
    (
        nb.types.List(nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:])))
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:, :])
    ),
    fastmath=True,
    nopython=True,
    looplift=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def clusterize_tasks(atoms: np.ndarray, links: np.ndarray, clusters_coords: np.ndarray) -> list:
    return [get_cluster_task_data(atoms, links, clusters_coords[i]) for i in nb.prange(clusters_coords.shape[0])]


@nb.jit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:])
    ),
    fastmath=True,
    nopython=True,
    looplift=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def interact_cluster(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, links: np.ndarray, cluster_mask: np.ndarray):
    coords_columns = np.array([A_COL_X, A_COL_Y])

    for i in nb.prange(cluster_atoms.shape[0]):
        atom = cluster_atoms[i]

        # исключим сам атом из соседей
        mask_exclude_self = ((neighbour_atoms[:, A_COL_X] != atom[A_COL_X])
                             | (neighbour_atoms[:, A_COL_Y] != atom[A_COL_Y]))
        neighbours = neighbour_atoms[mask_exclude_self]
        d = neighbours[:, coords_columns] - atom[coords_columns]
        l2 = d[:, 0] ** 2 + d[:, 1] ** 2
        l = np.sqrt(l2)

        # исключим слишком далекие атомы
        mask_in_radius = l <= CLUSTER_SIZE
        neighbours = neighbours[mask_in_radius]
        d = d[mask_in_radius]
        l = l[mask_in_radius]

        # возьмем атомы, не столкнувшиеся с данным
        mask_bounced = l < neighbours[:, A_COL_R] + atom[A_COL_R]
        neighbours_nb = neighbours[~mask_bounced]
        nb_d = d[~mask_bounced]
        nb_l = l[~mask_bounced]

        # получим множители гравитации согласно правилам взаимодействия частиц
        mult = ATOMS_GRAVITY[int(atom[A_COL_TYPE]), neighbours_nb[:, A_COL_TYPE].astype(np.int64)]

        # найдем ускорение за счет сил гравитации/антигравитации
        nb_nd = (nb_d.T / nb_l).T
        nb_dv = (nb_nd.T / nb_l).T  # l2 вместо l ???
        nb_dv = (nb_dv.T * mult).T
        nb_dv = np.sum(nb_dv, axis=0) * 3  # TODO factor

        # возьмем атомы, столкнувшиеся с данным
        b_d = d[mask_bounced]
        b_l = l[mask_bounced]

        # найдем ускорение за счет сил упругости
        b_nd = (b_d.T / b_l).T
        b_dv = (b_nd.T / np.maximum(b_l, 2)).T  # TODO factor
        b_dv = np.sum(b_dv, axis=0) * 0.5  # TODO factor

        # применим суммарное ускорение
        atom[A_COL_VX] += nb_dv[0] - b_dv[0]
        atom[A_COL_VY] += nb_dv[1] - b_dv[1]

        # получим связи атома
        links_mask = (links[:, L_COL_LHS] == int(atom[A_COL_ID])) | (links[:, L_COL_RHS] == int(atom[A_COL_ID]))
        atom_links = links[links_mask]

        # получим не связанных с атомом соседей
        mask_linked = (isin(neighbours[:, A_COL_ID], atom_links[:, L_COL_LHS])
                       | isin(neighbours[:, A_COL_ID], atom_links[:, L_COL_RHS]))
        not_linked_neighbours = neighbours[~mask_linked]
        nl_l = l[~mask_linked]

        # создадим новые связи с близкими атомами
        close_neighbours = not_linked_neighbours[nl_l < 30]
        new_links = np.zeros(shape=(close_neighbours.shape[0], 2), dtype=np.int64)
        new_links[:, 0] = np.repeat(atom[A_COL_ID], close_neighbours.shape[0]).astype(np.int64)
        new_links[:, 1] = close_neighbours[:, A_COL_ID].T.astype(np.int64)

    return cluster_atoms, cluster_mask


@nb.jit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:, :])
    ),
    fastmath=True,
    nopython=True,
    looplift=True,
    boundscheck=False,
    parallel=True,
    cache=not MODE_DEBUG,
)
def interact_all(atoms: np.ndarray, links: np.ndarray, clusters_coords: np.ndarray) -> None:
    tasks = clusterize_tasks(atoms, links, clusters_coords)
    for i in nb.prange(len(tasks)):
        task_data = tasks[i]
        cluster_atoms, cluster_mask = interact_cluster(*task_data)
        atoms[cluster_mask] = cluster_atoms


@nb.jit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:])
    ),
    fastmath=True,
    nopython=True,
    looplift=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def apply_speed(data: np.ndarray, max_coord: np.ndarray) -> None:
    data[:, A_COL_X] += data[:, A_COL_VX]
    data[:, A_COL_Y] += data[:, A_COL_VY]

    mask_x_min = data[:, A_COL_X] < 0
    mask_y_min = data[:, A_COL_Y] < 0
    mask_x_max = data[:, A_COL_X] > max_coord[0]
    mask_y_max = data[:, A_COL_Y] > max_coord[1]

    data[mask_x_min, A_COL_VX] *= -1
    data[mask_y_min, A_COL_VY] *= -1
    data[mask_x_min, A_COL_X] *= -1
    data[mask_y_min, A_COL_Y] *= -1

    data[mask_x_max, A_COL_VX] *= -1
    data[mask_y_max, A_COL_VY] *= -1
    data[mask_x_max, A_COL_X] = max_coord[0] - (data[mask_x_max, A_COL_X] - max_coord[0])
    data[mask_y_max, A_COL_Y] = max_coord[1] - (data[mask_y_max, A_COL_Y] - max_coord[1])

    data[:, A_COL_VX] *= 0.98  # TODO factor
    data[:, A_COL_VY] *= 0.98  # TODO factor

    data[:, A_COL_CX] = np.floor(data[:, A_COL_X] / CLUSTER_SIZE)
    data[:, A_COL_CY] = np.floor(data[:, A_COL_Y] / CLUSTER_SIZE)
