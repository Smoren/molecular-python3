import numpy as np
import numba as nb

from app.constants import A_COL_CX, A_COL_CY, A_COL_X, A_COL_Y, A_COL_VX, A_COL_VY, \
    A_COL_TYPE
from app.config import USE_JIT_CACHE


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
        nb.types.Tuple((nb.float64[:, :], nb.float64[:, :], nb.boolean[:]))
        (nb.float64[:, :], nb.float64[:])
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_cluster_task_data(atoms: np.ndarray, cluster_coords: np.ndarray) -> tuple:
    cluster_x, cluster_y = cluster_coords[0], cluster_coords[1]

    cluster_mask = (atoms[:, A_COL_CX] == cluster_x) & \
                   (atoms[:, A_COL_CY] == cluster_y)
    neighbours_mask = (atoms[:, A_COL_CX] >= cluster_x - 1) & \
                      (atoms[:, A_COL_CX] <= cluster_x + 1) & \
                      (atoms[:, A_COL_CY] >= cluster_y - 1) & \
                      (atoms[:, A_COL_CY] <= cluster_y + 1)

    cluster_atoms, neighbours_atoms = atoms[cluster_mask], atoms[neighbours_mask]

    return cluster_atoms, neighbours_atoms, cluster_mask


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
    atoms_lj_params: np.ndarray,
    force_gravity: float,
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
        neighbours_d = _d[_mask_in_radius]
        neighbours_l = _l[_mask_in_radius]
        ###############################

        # [Найдем ускорение гравитационных взаимодействий атома с не связанными соседями]
        _d_norm = (neighbours_d.T / neighbours_l).T

        sigma, eps = atoms_lj_params[int(atom[A_COL_TYPE])]

        _ljp = -lennard_jones_potential(neighbours_l/5, sigma, eps)  # -2R
        _ljp[_ljp < -1.5] = -1.5
        _f = (_d_norm.T * _ljp).T

        ###############################
        dv_gravity = np.sum(_f, axis=0) * force_gravity
        ###############################

        # [Применим ускорение]
        atom[A_COL_VX] += dv_gravity[0]
        atom[A_COL_VY] += dv_gravity[1]

    return cluster_atoms, cluster_mask


@nb.njit(
    (
        nb.types.NoneType('none')
        (
            nb.float64[:, :], nb.float64[:, :],
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
    clusters_coords: np.ndarray,
    cluster_size: int,
    atoms_lj_params: np.ndarray,
    force_not_linked_gravity: float,
) -> None:
    for i in nb.prange(clusters_coords.shape[0]):
        task_data = get_cluster_task_data(atoms, clusters_coords[i])
        cluster_atoms, neighbours_atoms, cluster_mask = task_data
        cluster_atoms, cluster_mask = interact_cluster(
            cluster_atoms, neighbours_atoms, cluster_mask,
            cluster_size, atoms_lj_params, force_not_linked_gravity,
        )
        atoms[cluster_mask] = cluster_atoms


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
