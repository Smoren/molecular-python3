from typing import Callable

import torch

from app.config import DEVICE


def np_apply_reducer(arr: torch.Tensor, func1d: Callable, axis: int) -> torch.Tensor:
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = torch.empty(arr.shape[1]).to(DEVICE)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = torch.empty(arr.shape[0]).to(DEVICE)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


def np_unique_links(arr: torch.Tensor) -> torch.Tensor:
    assert arr.ndim == 2

    if arr.shape[0] == 0:
        return arr

    result = set()
    for i in range(arr.shape[0]):
        result.add((arr[i, 0], arr[i, 1], arr[i, 2]))

    return torch.tensor(list(result)).to(DEVICE)
