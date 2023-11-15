from multiprocessing import shared_memory
from typing import Tuple

import numpy as np


def create_shared_variable_for_cluster(
    cluster_coords: Tuple[int, ...],
    data: np.ndarray,
    name_prefix: str,
) -> shared_memory.SharedMemory:
    cluster_name = f'{name_prefix}_{str(cluster_coords)}'
    shared_var_size = np.dtype(np.float64).itemsize * np.prod(data.shape)

    try:
        shm = shared_memory.SharedMemory(create=True, size=shared_var_size, name=cluster_name)
    except FileExistsError:
        destroy_shared_variable_by_name(cluster_name)
        shm = shared_memory.SharedMemory(create=True, size=shared_var_size, name=cluster_name)
    dst = np.ndarray(shape=data.shape, dtype=np.float64, buffer=shm.buf)
    dst[:] = data[:]

    return shm


def get_shared_variable(
    cluster_coords: Tuple[int, ...],
    shape: Tuple[int, ...],
    name_prefix: str,
) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    cluster_name = f'{name_prefix}_{str(cluster_coords)}'

    shm = shared_memory.SharedMemory(name=cluster_name)
    data = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

    return data, shm


def destroy_shared_variable_by_name(name: str):
    shm = shared_memory.SharedMemory(name=name)
    destroy_shared_variable(shm)


def destroy_shared_variable(shm: shared_memory.SharedMemory):
    shm.close()
    shm.unlink()
