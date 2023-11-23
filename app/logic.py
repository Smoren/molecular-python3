import numpy as np
import numba as nb

from app.constants import A_COL_CX, A_COL_CY, A_COL_X, A_COL_Y, A_COL_VX, A_COL_VY, \
    A_COL_TYPE, A_COL_R, A_COL_ID, L_COL_LHS, L_COL_RHS, A_COL_LINKS, L_COL_DEL
from app.config import USE_JIT_CACHE
from app.utils import isin, np_apply_reducer, concat, np_unique_links


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def lennard_jones_potential(r, sigma: float, eps: float) -> np.ndarray:
    buf = (sigma / r)**6
    return 4 * eps * (buf * buf - buf)


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def lennard_jones_potential_truncated(r: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    rc = 2.5*sigma
    result = lennard_jones_potential(r, sigma, eps) - lennard_jones_potential(np.array([rc]), sigma, eps)
    result[r > rc] = 0
    return result


@nb.njit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.int64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.int64[:, :], nb.float64[:])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_cluster_task_data(atoms: np.ndarray, links: np.ndarray, cluster_coords: np.ndarray) -> tuple:
    cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

    cluster_mask = (atoms[:, A_COL_CX] == cluster_x) & \
                   (atoms[:, A_COL_CY] == cluster_y)
    neighbours_mask = (atoms[:, A_COL_CX] >= cluster_x - 1) & \
                      (atoms[:, A_COL_CX] <= cluster_x + 1) & \
                      (atoms[:, A_COL_CY] >= cluster_y - 1) & \
                      (atoms[:, A_COL_CY] <= cluster_y + 1)

    cluster_atoms, neighbours_atoms = atoms[cluster_mask], atoms[neighbours_mask]

    # TODO узкое место
    mask_links = (isin(links[:, L_COL_LHS], cluster_atoms[:, A_COL_ID].astype(np.int64)) |
                  isin(links[:, L_COL_RHS], cluster_atoms[:, A_COL_ID].astype(np.int64)))
    links_filtered = links[mask_links]
    # links_filtered = links

    return cluster_atoms, neighbours_atoms, links_filtered, cluster_mask


@nb.njit(
    (
        nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))
        (
            nb.float64[:, :], nb.float64[:, :], nb.boolean[:],
            nb.int64, nb.float64[:, :], nb.float64,
        )
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def interact_cluster(
    cluster_atoms: np.ndarray,
    neighbour_atoms: np.ndarray,
    cluster_mask: np.ndarray,
    cluster_size: int,
    atoms_gravity: np.ndarray,
    force_not_linked_gravity: float,
):
    coords_columns = np.array([A_COL_X, A_COL_Y])

    for i in nb.prange(cluster_atoms.shape[0]):
        atom = cluster_atoms[i]

        # [Определим соседей, с которыми возможно взаимодействие]
        _mask_exclude_self = ((neighbour_atoms[:, A_COL_X] != atom[A_COL_X]) |
                              (neighbour_atoms[:, A_COL_Y] != atom[A_COL_Y]))
        neighbours = neighbour_atoms[_mask_exclude_self]
        _d = neighbours[:, coords_columns] - atom[coords_columns]
        _l2 = _d[:, 0] ** 2 + _d[:, 1] ** 2
        _l = np.sqrt(_l2)
        _mask_in_radius = _l <= cluster_size

        ###############################
        neighbours = neighbours[_mask_in_radius]
        neighbours_d = _d[_mask_in_radius]
        neighbours_l = _l[_mask_in_radius]
        ###############################

        # [Найдем ускорение гравитационных взаимодействий атома с не связанными соседями]
        _mult = atoms_gravity[int(atom[A_COL_TYPE]), neighbours[:, A_COL_TYPE].astype(np.int64)]
        _d_norm = (neighbours_d.T / neighbours_l).T
        _m1 = np.pi * atom[A_COL_R] ** 2
        _m2 = np.pi * (neighbours[:, A_COL_R].T ** 2).T

        _ljp = -lennard_jones_potential(neighbours_l/5, 3, 0.01)  # -2R
        _ljp[_ljp < -1.5] = -1.5
        _f = (_d_norm.T * _ljp).T

        ###############################
        dv_gravity = np.sum((_f.T * _mult).T, axis=0) * force_not_linked_gravity
        ###############################

        # [Применим ускорение]
        atom[A_COL_VX] += dv_gravity[0]
        atom[A_COL_VY] += dv_gravity[1]

    return cluster_atoms, cluster_mask


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:, :], nb.int64[:], nb.int64[:, :])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def handle_new_links(
    atoms: np.ndarray,
    links: np.ndarray,
    atoms_links: np.ndarray,
    atoms_link_types: np.ndarray,
):
    for i in nb.prange(links.shape[0]):
        link_candidate = links[i]

        lhs_type = int(atoms[link_candidate[L_COL_LHS], A_COL_TYPE])
        rhs_type = int(atoms[link_candidate[L_COL_RHS], A_COL_TYPE])

        lhs_total_links = atoms[link_candidate[L_COL_LHS], A_COL_LINKS]
        rhs_total_links = atoms[link_candidate[L_COL_RHS], A_COL_LINKS]
        lhs_type_links = atoms[link_candidate[L_COL_LHS], A_COL_LINKS+1+rhs_type]
        rhs_type_links = atoms[link_candidate[L_COL_RHS], A_COL_LINKS+1+lhs_type]

        lhs_total_links_max = atoms_links[lhs_type]
        rhs_total_links_max = atoms_links[rhs_type]
        lhs_type_links_max = atoms_link_types[lhs_type][rhs_type]
        rhs_type_links_max = atoms_link_types[rhs_type][lhs_type]

        can_link = lhs_total_links < lhs_total_links_max and \
            rhs_total_links < rhs_total_links_max and \
            lhs_type_links < lhs_type_links_max and \
            rhs_type_links < rhs_type_links_max
        link_candidate[L_COL_DEL] = not can_link
        link_plus = int(can_link)

        atoms[link_candidate[L_COL_LHS], A_COL_LINKS] += link_plus
        atoms[link_candidate[L_COL_RHS], A_COL_LINKS] += link_plus

        atoms[link_candidate[L_COL_LHS], int(A_COL_LINKS+1+rhs_type)] += link_plus
        atoms[link_candidate[L_COL_RHS], int(A_COL_LINKS+1+lhs_type)] += link_plus


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:, :])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def handle_deleting_links(atoms: np.ndarray, links: np.ndarray) -> None:
    for i in nb.prange(links.shape[0]):
        link = links[i]

        lhs_type = int(atoms[link[L_COL_LHS], A_COL_TYPE])
        rhs_type = int(atoms[link[L_COL_RHS], A_COL_TYPE])

        atoms[link[L_COL_LHS], A_COL_LINKS] -= 1
        atoms[link[L_COL_RHS], A_COL_LINKS] -= 1

        atoms[link[L_COL_LHS], A_COL_LINKS+1+rhs_type] -= 1
        atoms[link[L_COL_RHS], A_COL_LINKS+1+lhs_type] -= 1


@nb.njit(
    (
        nb.types.NoneType('none')
        (
            nb.float64[:, :], nb.int64[:, :], nb.float64[:, :],
            nb.int64, nb.float64[:, :], nb.float64,
        )
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    parallel=True,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def interact_atoms(
    atoms: np.ndarray,
    links: np.ndarray,
    clusters_coords: np.ndarray,
    cluster_size: int,
    atoms_gravity: np.ndarray,
    force_not_linked_gravity: float,
) -> None:
    for i in nb.prange(clusters_coords.shape[0]):
        task_data = get_cluster_task_data(atoms, links, clusters_coords[i])
        cluster_atoms, neighbours_atoms, links_filtered, cluster_mask = task_data
        cluster_atoms, cluster_mask = interact_cluster(
            cluster_atoms, neighbours_atoms, cluster_mask,
            cluster_size, atoms_gravity, force_not_linked_gravity,
        )
        atoms[cluster_mask] = cluster_atoms


@nb.njit(
    (
        nb.int64[:, :]
        (nb.float64[:, :], nb.int64[:, :], nb.float64)
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def interact_links(atoms: np.ndarray, links: np.ndarray, max_link_distance: float) -> np.ndarray:
    lhs_atoms = atoms[links[:, L_COL_LHS]]
    rhs_atoms = atoms[links[:, L_COL_RHS]]

    coords_columns = np.array([A_COL_X, A_COL_Y])
    d = rhs_atoms[:, coords_columns] - lhs_atoms[:, coords_columns]
    l2 = d[:, 0]**2 + d[:, 1]**2
    l = np.sqrt(l2)

    filter_mask = l < max_link_distance
    links_to_save = links[filter_mask]
    links_to_delete = links[~filter_mask]

    handle_deleting_links(atoms, links_to_delete)

    return links_to_save


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64[:], nb.int64, nb.float64, nb.float64)
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def apply_speed(
    atoms: np.ndarray,
    max_coord: np.ndarray,
    cluster_size: int,
    inertial_factor: float,
    simulation_speed: float,
) -> None:
    atoms[:, A_COL_X] += atoms[:, A_COL_VX] * simulation_speed
    atoms[:, A_COL_Y] += atoms[:, A_COL_VY] * simulation_speed

    mask_x_min = atoms[:, A_COL_X] < 0
    mask_y_min = atoms[:, A_COL_Y] < 0
    mask_x_max = atoms[:, A_COL_X] > max_coord[0]
    mask_y_max = atoms[:, A_COL_Y] > max_coord[1]

    atoms[mask_x_min, A_COL_VX] *= -1
    atoms[mask_y_min, A_COL_VY] *= -1
    atoms[mask_x_min, A_COL_X] *= -1
    atoms[mask_y_min, A_COL_Y] *= -1

    atoms[mask_x_max, A_COL_VX] *= -1
    atoms[mask_y_max, A_COL_VY] *= -1
    atoms[mask_x_max, A_COL_X] = max_coord[0] - (atoms[mask_x_max, A_COL_X] - max_coord[0])
    atoms[mask_y_max, A_COL_Y] = max_coord[1] - (atoms[mask_y_max, A_COL_Y] - max_coord[1])

    atoms[:, A_COL_VX] *= inertial_factor
    atoms[:, A_COL_VY] *= inertial_factor

    atoms[:, A_COL_CX] = np.floor(atoms[:, A_COL_X] / cluster_size)
    atoms[:, A_COL_CY] = np.floor(atoms[:, A_COL_Y] / cluster_size)
