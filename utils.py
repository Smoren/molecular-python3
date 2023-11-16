import math
import time

import numpy as np
import numba as nb


@nb.jit(
    (nb.types.Tuple((nb.float64, nb.float64))(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32)),
    fastmath=True,
    nopython=True
)
def calc_speed_delta(
    lhs_x: int, rhs_x: int, lhs_y: int, rhs_y: int, lhs_radius: int, rhs_radius: int
):
    dx, dy = lhs_x - rhs_x, lhs_y - rhs_y
    dist = math.sqrt(dx*dx + dy*dy)
    radius_sum = lhs_radius + rhs_radius
    rhs_mass = math.pi * rhs_radius * rhs_radius

    if dist < radius_sum:
        delta_x = -rhs_x + lhs_x
        delta_y = -rhs_y + lhs_y
        a = 0.01 * rhs_mass / dist / dist if dist > 0 else 0

        delta_vx = delta_x * (radius_sum - dist) * a
        delta_vy = delta_y * (radius_sum - dist) * a
    else:
        delta_x = rhs_x - lhs_x
        delta_y = rhs_y - lhs_y
        a = 2 * rhs_mass / dist / dist if dist > 0 else 0

        delta_vx = delta_x / dist * a if dist > 0 else 0
        delta_vy = delta_y / dist * a if dist > 0 else 0

    return delta_vx, delta_vy


def randomize_args(n: int) -> np.ndarray:
    result = np.array([
        np.random.randint(low=0, high=100, size=n),
        np.random.randint(low=0, high=100, size=n),
        np.random.randint(low=0, high=100, size=n),
        np.random.randint(low=0, high=100, size=n),
        np.random.randint(low=0, high=10, size=n),
        np.random.randint(low=0, high=10, size=n),
    ]).T

    return result


def bench():
    args_list = randomize_args(1000000)

    ts = time.time()
    for args in args_list:
        delta_vx, delta_vy = calc_speed_delta(*args)
    print(time.time() - ts)


if __name__ == "__main__":
    bench()
