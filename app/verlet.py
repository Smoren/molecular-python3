import numpy as np
import numba as nb

from app.config import USE_JIT_CACHE
from app.constants import A_COL_VX, A_COL_FX, A_COL_R, A_COL_VY, A_COL_FY


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def morse_potential(r: np.ndarray, eps: float, alpha: float, sigma: float):
    return eps * (np.exp(-2*alpha * (r - sigma)) - 2*np.exp(-alpha * (r - sigma)))


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def morse_force(r: np.ndarray, eps: float, alpha: float, sigma: float):
    return -2*alpha * eps * np.exp(alpha*(sigma-r)) * (np.exp(alpha*(sigma-r)) - 1)


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_kinetic_energy(m, v):
    return m * (v ** 2) / 2


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_verlet_next_x(x, v, f, m, dt):
    a = f / m
    return x + v * dt + a * (dt ** 2) / 2


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_verlet_next_f(r, rx, eps: float, alpha: float, sigma: float):
    return np.sum(morse_force(r, eps, alpha, sigma) * rx / r)


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_verlet_next_u(rs, eps: float, alpha: float, sigma: float):
    return np.sum(morse_potential(rs, eps, alpha, sigma)) / 2


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_verlet_next_v(v, f, f_next, m, dt):
    a, a_next = f / m, f_next / m
    return v + ((a + a_next) * dt) / 2


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def get_verlet_next(p: np.ndarray, rx: np.ndarray, ry: np.ndarray, dt: float, eps: float, alpha: float, sigma: float):
    r = np.sqrt(np.power(rx, 2) + np.power(ry, 2))
    m = np.pi * np.power(p[A_COL_R], 2)

    fx_next = get_verlet_next_f(r, rx, eps, alpha, sigma)
    fy_next = get_verlet_next_f(r, ry, eps, alpha, sigma)

    vx_next = get_verlet_next_v(p[A_COL_VX], p[A_COL_FX], fx_next, m, dt)
    vy_next = get_verlet_next_v(p[A_COL_VY], p[A_COL_FY], fy_next, m, dt)
    v_next = np.sqrt(np.power(vx_next, 2) + np.power(vy_next, 2))

    u_next = get_verlet_next_u(r, eps, alpha, sigma)
    ek_next = get_kinetic_energy(m, v_next)

    return fx_next, fy_next, vx_next, vy_next, v_next, u_next, ek_next
