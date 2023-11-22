from typing import List, Callable

import numpy as np
import numba as nb

from app.config import USE_JIT_CACHE


@nb.njit(
    fastmath=True,
    boundscheck=False,
    looplift=True,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def isin(where: np.ndarray, what: np.ndarray) -> np.ndarray:
    where_size = where.shape[0]
    what_size = what.shape[0]
    result = np.empty(shape=(where_size,), dtype=np.bool_)
    for i in nb.prange(where_size):
        result[i] = False

    if what_size == 0:
        return result

    what_min, what_max = what[0], what[0]
    for i in nb.prange(1, what_size):
        if what[i] > what_max:
            what_max = what[i]
        elif what[i] < what_min:
            what_min = what[i]

    what_range = what_max - what_min

    what_normalized = np.empty(shape=(what_size+1,), dtype=np.int64)
    for i in nb.prange(what_size):
        what_normalized[i] = what[i] - what_min

    isin_helper_ar = np.empty(shape=(what_range+1,), dtype=np.int64)
    for i in nb.prange(what_range+1):
        isin_helper_ar[i] = False
    for i in nb.prange(what_size):
        isin_helper_ar[what_normalized[i]] = True

    for i in nb.prange(where_size):
        if where[i] > what_max or where[i] < what_min:
            continue
        result[i] = isin_helper_ar[where[i] - what_min]

    return result


@nb.njit(
    fastmath=True,
    boundscheck=False,
    looplift=True,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def isin_old(a, b):
    out = np.empty(a.shape[0], dtype=np.bool_)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        out[i] = a[i] in b
    return out


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
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
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
)
def np_unique_links(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2

    if arr.shape[0] == 0:
        return arr

    result = set()
    for i in nb.prange(arr.shape[0]):
        result.add((arr[i, 0], arr[i, 1], arr[i, 2]))
    return np.array(list(result))


@nb.njit(
    fastmath=True,
    looplift=True,
    boundscheck=False,
    nogil=True,
    cache=USE_JIT_CACHE,
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
