from typing import List, Callable

import numpy as np
import numba as nb

from app.constants import A_COL_CX, A_COL_CY, A_COL_X, A_COL_Y, A_COL_VX, A_COL_VY, A_COL_TYPE, A_COL_R, A_COL_ID, \
    L_COL_LHS, L_COL_RHS
from app.config import ATOMS_GRAVITY, CLUSTER_SIZE, MODE_DEBUG, ATOMS_LINKS, ATOMS_LINK_GRAVITY


@nb.njit(
    fastmath=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def np_apply_reducer(arr: np.ndarray, func1d: Callable, axis: int) -> np.ndarray:
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in nb.prange(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in nb.prange(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit(
    fastmath=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def np_unique_links(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2

    if arr.shape[0] == 0:
        return arr

    result = set()
    for i in nb.prange(arr.shape[0]):
        result.add((arr[i, 0], arr[i, 1]))
    return np.array(list(result))


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


@nb.njit(
    fastmath=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def concat(arrays: List[np.ndarray], columns_count: int, dtype: np.dtype) -> np.ndarray:
    total_len = 0
    for i in nb.prange(len(arrays)):
        total_len += arrays[i].shape[0]

    result = np.empty(shape=(total_len, columns_count), dtype=dtype)

    k = 0
    for i in nb.prange(len(arrays)):
        for j in nb.prange(arrays[i].shape[0]):
            result[k] = arrays[i][j]
            k += 1

    return result


@nb.njit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:])
    ),
    fastmath=True,
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


@nb.njit(
    (
        nb.types.List(nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:])))
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:, :])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def clusterize_tasks(atoms: np.ndarray, links: np.ndarray, clusters_coords: np.ndarray) -> list:
    return [get_cluster_task_data(atoms, links, clusters_coords[i]) for i in nb.prange(clusters_coords.shape[0])]


@nb.njit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.int64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    cache=not MODE_DEBUG,
)
def interact_cluster(cluster_atoms: np.ndarray, neighbour_atoms: np.ndarray, links: np.ndarray, cluster_mask: np.ndarray):
    coords_columns = np.array([A_COL_X, A_COL_Y])
    new_links = nb.typed.List.empty_list(nb.int64[:, :])

    for i in nb.prange(cluster_atoms.shape[0]):
        atom = cluster_atoms[i]

        # [Определим соседей, с которыми возможно взаимодействие]
        _mask_exclude_self = ((neighbour_atoms[:, A_COL_X] != atom[A_COL_X]) |
                              (neighbour_atoms[:, A_COL_Y] != atom[A_COL_Y]))
        neighbours = neighbour_atoms[_mask_exclude_self]
        _d = neighbours[:, coords_columns] - atom[coords_columns]
        _l2 = _d[:, 0] ** 2 + _d[:, 1] ** 2
        _l = np.sqrt(_l2)
        _mask_in_radius = _l <= CLUSTER_SIZE

        ###############################
        neighbours = neighbours[_mask_in_radius]
        neighbours_d = _d[_mask_in_radius]
        neighbours_l = _l[_mask_in_radius]
        ###############################

        # [Получим связи атома]
        _links_mask = (links[:, L_COL_LHS] == int(atom[A_COL_ID])) | (links[:, L_COL_RHS] == int(atom[A_COL_ID]))

        ###############################
        atom_links = links[_links_mask]
        ###############################

        # [Разделим соседей на столкнувшихся и не столкнувшихся с атомом]
        _mask_bounced = neighbours_l < neighbours[:, A_COL_R] + atom[A_COL_R]

        ###############################
        neighbours_bounced_d = neighbours_d[_mask_bounced]
        neighbours_bounced_l = neighbours_l[_mask_bounced]
        ###############################

        ###############################
        neighbours_not_bounced = neighbours[~_mask_bounced]
        neighbours_not_bounced_d = neighbours_d[~_mask_bounced]
        neighbours_not_bounced_l = neighbours_l[~_mask_bounced]
        ###############################

        # [Разделим не столкнувшихся соседей на связанные и не связанные с атомом]
        _mask_linked = (isin(neighbours_not_bounced[:, A_COL_ID], atom_links[:, L_COL_LHS]) |
                        isin(neighbours_not_bounced[:, A_COL_ID], atom_links[:, L_COL_RHS]))

        ###############################
        neighbours_linked = neighbours_not_bounced[_mask_linked]
        neighbours_linked_d = neighbours_not_bounced_d[_mask_linked]
        neighbours_linked_l = neighbours_not_bounced_l[_mask_linked]
        ###############################

        ###############################
        neighbours_not_linked = neighbours_not_bounced[~_mask_linked]
        neighbours_not_linked_d = neighbours_not_bounced_d[~_mask_linked]
        neighbours_not_linked_l = neighbours_not_bounced_l[~_mask_linked]
        ###############################

        # [Из не связанных с атомом соседей найдем те, с которыми построим новые связи]
        _mask_to_link = neighbours_not_linked_l < 15  # TODO factor

        ###############################
        neighbours_to_link = neighbours_not_linked[_mask_to_link]
        ###############################

        # [Найдем ускорение отталкивания атома от столкнувшихся с ним соседей]
        _d_norm = (neighbours_bounced_d.T / neighbours_bounced_l).T
        _l_limited = np.maximum(neighbours_bounced_l, 2)  # TODO factor

        ###############################
        dv_elastic = -np.sum((_d_norm.T / _l_limited).T, axis=0) * 0.5  # TODO factor
        ###############################

        # [Найдем ускорение гравитационных взаимодействий атома с не связанными соседями]
        _mult = ATOMS_GRAVITY[int(atom[A_COL_TYPE]), neighbours_not_linked[:, A_COL_TYPE].astype(np.int64)]
        _d_norm = (neighbours_not_linked_d.T / neighbours_not_linked_l).T
        _f = (_d_norm.T / neighbours_not_linked_l).T  # l2 вместо l ???

        ###############################
        dv_gravity_not_linked = np.sum((_f.T * _mult).T, axis=0) * 0.5  # TODO factor
        ###############################

        # [Найдем ускорение гравитационных и связных взаимодействий атома со связанными соседями]
        _mult = ATOMS_LINK_GRAVITY[int(atom[A_COL_TYPE]), neighbours_linked[:, A_COL_TYPE].astype(np.int64)]
        _d_norm = (neighbours_linked_d.T / neighbours_linked_l).T
        _f = (_d_norm.T * neighbours_linked_l).T  # l2 вместо l ???

        ###############################
        dv_gravity_linked = np.sum((_f.T * _mult).T, axis=0) * 0.03  # TODO factor
        dv_link_influence = np.sum(_d_norm, axis=0) * 0.3  # TODO factor
        ###############################

        # [Применим суммарное ускорение]
        atom[A_COL_VX] += dv_elastic[0] + dv_gravity_not_linked[0] + dv_link_influence[0] + dv_gravity_linked[0]
        atom[A_COL_VY] += dv_elastic[1] + dv_gravity_not_linked[1] + dv_link_influence[1] + dv_gravity_linked[1]

        # [Если связи заполнены, делать больше нечего]
        max_atom_links = ATOMS_LINKS[int(atom[A_COL_TYPE])]
        if atom_links.shape[0] > max_atom_links:
            continue

        # [Создаем новые связи]
        new_atom_links = np.empty(shape=(neighbours_to_link.shape[0], 2), dtype=np.int64)
        new_atom_links[:, 0] = np.repeat(atom[A_COL_ID], neighbours_to_link.shape[0]).astype(np.int64)
        new_atom_links[:, 1] = neighbours_to_link[:, A_COL_ID].T.astype(np.int64)

        # [Если кандидаты есть]
        if new_atom_links.shape[0] > 0:
            # [Отсортируем ID в кортежах связей по возрастанию]
            new_atom_links[:, 0], new_atom_links[:, 1] = np_apply_reducer(
                new_atom_links, np.min, axis=1,
            ), np_apply_reducer(
                new_atom_links, np.max, axis=1,
            )

            # [Ограничим количество новых связей общим лимитом и добавим в выборку]
            new_atom_links = new_atom_links[:(max_atom_links-atom_links.shape[0])]
            new_links.append(new_atom_links)

    new_links_total = np_unique_links(concat(new_links, links.shape[1], np.int64))

    return cluster_atoms, new_links_total, cluster_mask


@nb.njit(
    (
        nb.int64[:, :]
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:, :])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    parallel=True,
    cache=not MODE_DEBUG,
)
def interact_atoms(atoms: np.ndarray, links: np.ndarray, clusters_coords: np.ndarray) -> np.ndarray:
    tasks = clusterize_tasks(atoms, links, clusters_coords)
    new_links = [np.empty(shape=(0, 2), dtype=np.int64)] * len(tasks)
    for i in nb.prange(len(tasks)):
        task_data = tasks[i]
        cluster_atoms, cluster_new_links, cluster_mask = interact_cluster(*task_data)
        atoms[cluster_mask] = cluster_atoms
        new_links[i] = np.empty(shape=(cluster_new_links.shape[0], 2), dtype=np.int64)
        for j in nb.prange(cluster_new_links.shape[0]):
            new_links[i][j] = cluster_new_links[j]

    total_new_links = np_unique_links(concat(new_links, links.shape[1], np.int64))

    if len(total_new_links) > 0:
        print(f'new links: {len(total_new_links)}')

    return total_new_links


@nb.njit(
    (
        nb.int64[:, :]
        (nb.float64[:, :], nb.int64[:, :])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    # parallel=True,
    cache=not MODE_DEBUG,
)
def interact_links(atoms: np.ndarray, links: np.ndarray) -> np.ndarray:
    lhs_atoms = atoms[links[:, L_COL_LHS]]
    rhs_atoms = atoms[links[:, L_COL_RHS]]

    coords_columns = np.array([A_COL_X, A_COL_Y])
    d = rhs_atoms[:, coords_columns] - lhs_atoms[:, coords_columns]
    l2 = d[:, 0]**2 + d[:, 1]**2
    l = np.sqrt(l2)

    filter_mask = l < 25  # TODO factor
    links = links[filter_mask]
    lhs_atoms = atoms[links[:, L_COL_LHS]]
    rhs_atoms = atoms[links[:, L_COL_RHS]]

    d = d[filter_mask]
    l = l[filter_mask]

    nd = (d.T / l).T
    dv = (nd.T / l).T  # l2 вместо l ???
    # dv = (dv.T * mult).T
    dv = np.sum(dv, axis=0) * 3  # TODO factor

    lhs_atoms[:, A_COL_VX] += dv[0]
    lhs_atoms[:, A_COL_VY] += dv[1]
    rhs_atoms[:, A_COL_VX] -= dv[0]
    rhs_atoms[:, A_COL_VY] -= dv[1]

    return links


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:])
    ),
    fastmath=True,
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
