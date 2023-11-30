import numpy as np
import numba as nb

from app.constants import A_COL_CX, A_COL_CY, A_COL_X, A_COL_Y, A_COL_VX, A_COL_VY, \
    A_COL_TYPE, A_COL_X_NEXT, A_COL_Y_NEXT, A_COL_FX, A_COL_R, A_COL_FY, A_COL_VX_NEXT, A_COL_VY_NEXT, A_COL_FX_NEXT, \
    A_COL_FY_NEXT, A_COL_FIXED
from app.config import USE_JIT_CACHE
from app.verlet import get_verlet_next_x, get_verlet_next


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
                   (atoms[:, A_COL_CY] == cluster_y) & \
                   (atoms[:, A_COL_FIXED] < 1)
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
            nb.int64, nb.float64[:, :], nb.float64, nb.float64,
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
    gravity: float,
    delta_t: float,
):
    coords_columns = np.array([A_COL_X, A_COL_Y])

    for i in nb.prange(cluster_atoms.shape[0]):
        atom = cluster_atoms[i]

        _mask_exclude_self = ((neighbour_atoms[:, A_COL_X] != atom[A_COL_X]) |
                              (neighbour_atoms[:, A_COL_Y] != atom[A_COL_Y]))
        _neighbours = neighbour_atoms[_mask_exclude_self]
        _d = _neighbours[:, coords_columns] - atom[coords_columns]
        _l2 = _d[:, 0] ** 2 + _d[:, 1] ** 2
        _l = np.sqrt(_l2)
        _mask_in_radius = _l <= cluster_size
        neighbours = _neighbours[_mask_in_radius]

        rx = neighbours[:, A_COL_X_NEXT] - atom[A_COL_X_NEXT]
        ry = neighbours[:, A_COL_Y_NEXT] - atom[A_COL_Y_NEXT]

        eps, alpha, sigma = atoms_lj_params[int(atom[A_COL_TYPE])]
        eps = (eps + atoms_lj_params[neighbours[:, A_COL_TYPE].astype(np.int64)][:, 0]) / 2
        alpha = (alpha + atoms_lj_params[neighbours[:, A_COL_TYPE].astype(np.int64)][:, 1]) / 2
        sigma = (sigma + atoms_lj_params[neighbours[:, A_COL_TYPE].astype(np.int64)][:, 2]) / 2
        dt = delta_t

        fx_next, fy_next, vx_next, vy_next, v_next, u_next, ek_next \
            = get_verlet_next(atom, rx, ry, dt, eps, alpha, sigma, gravity)

        atom[A_COL_VX_NEXT] = vx_next
        atom[A_COL_VY_NEXT] = vy_next
        atom[A_COL_FX_NEXT] = fx_next
        atom[A_COL_FY_NEXT] = fy_next

    return cluster_atoms, cluster_mask


@nb.njit(
    (
        nb.types.NoneType('none')
        (
            nb.float64[:, :], nb.float64[:, :],
            nb.int64, nb.float64[:, :], nb.float64, nb.float64,
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
    gravity: float,
    delta_t: float,
) -> None:
    for i in nb.prange(clusters_coords.shape[0]):
        task_data = get_cluster_task_data(atoms, clusters_coords[i])
        cluster_atoms, neighbours_atoms, cluster_mask = task_data
        cluster_atoms, cluster_mask = interact_cluster(
            cluster_atoms, neighbours_atoms, cluster_mask,
            cluster_size, atoms_lj_params, gravity, delta_t,
        )
        atoms[cluster_mask] = cluster_atoms


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.float64)
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def calc_next_positions(atoms: np.ndarray, dt: float):
    m = np.pi * atoms[:, A_COL_R]**2
    atoms[:, A_COL_X_NEXT] = get_verlet_next_x(atoms[:, A_COL_X], atoms[:, A_COL_VX], atoms[:, A_COL_FX], m, dt)
    atoms[:, A_COL_Y_NEXT] = get_verlet_next_x(atoms[:, A_COL_Y], atoms[:, A_COL_VY], atoms[:, A_COL_FY], m, dt)


@nb.njit(
    (
        nb.types.NoneType('none')
        (nb.float64[:, :], nb.int64, nb.float64)
    ),
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def apply_next_values(atoms: np.ndarray, cluster_size: int, inertia: float) -> None:
    atoms[:, A_COL_X] = atoms[:, A_COL_X_NEXT]
    atoms[:, A_COL_Y] = atoms[:, A_COL_Y_NEXT]

    atoms[:, A_COL_VX] = atoms[:, A_COL_VX_NEXT] * inertia
    atoms[:, A_COL_VY] = atoms[:, A_COL_VY_NEXT] * inertia

    atoms[:, A_COL_FX] = atoms[:, A_COL_FX_NEXT]
    atoms[:, A_COL_FY] = atoms[:, A_COL_FY_NEXT]

    atoms[:, A_COL_CX] = np.floor(atoms[:, A_COL_X] / cluster_size)
    atoms[:, A_COL_CY] = np.floor(atoms[:, A_COL_Y] / cluster_size)


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
