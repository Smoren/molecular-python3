import time
from typing import Tuple

import numpy as np
import pygame
import numba as nb

from app.config import CONF_COLORS
from app.constants import COL_R, COL_Y, COL_X, COL_CX, COL_CY, COL_TYPE
from app.drawer import Drawer
from app.utils import interact_all, apply_speed


class Simulation:
    _screen: pygame.Surface
    _drawer: Drawer
    _clock: pygame.time.Clock
    _is_stopped: bool = False
    _atoms: np.ndarray
    _max_coord: np.ndarray
    _cluster_size: int

    def __init__(self, atoms: np.ndarray, window_size: Tuple[int, int], max_coord: Tuple[int, int], cluster_size: int):
        self._atoms = atoms
        self._max_coord = np.array(max_coord)
        self._cluster_size = cluster_size
        self._screen = pygame.display.set_mode(window_size)
        self._drawer = Drawer(self._screen)
        self._clock = pygame.time.Clock()

    def start(self):
        self._is_stopped = False
        while not self._is_stopped:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()

            ts = time.time_ns()
            self._step_move()
            self._step_interact()
            self._step_display()
            self._clock.tick(30)
            print(f'step spent: {(time.time_ns() - ts) / 1_000_000}')

    def stop(self):
        self._is_stopped = True

    def _step_interact(self) -> None:
        clusters_coords = np.unique(self._atoms[:, [COL_CX, COL_CY]], axis=0)
        interact_all(self._atoms, clusters_coords)

    def _step_move(self) -> None:
        apply_speed(self._atoms, self._max_coord, self._cluster_size)

    @nb.jit(
        forceobj=True,
        fastmath=True,
        looplift=True,
        boundscheck=False,
        parallel=True,
    )
    def _step_display(self) -> None:
        self._drawer.clear()

        for i in nb.prange(self._atoms.shape[0]):
            row = self._atoms[i]
            self._drawer.draw_circle((row[COL_X], row[COL_Y]), row[COL_R], CONF_COLORS[int(row[COL_TYPE])])
            i += 1
            if i > 10000:
                break

        self._drawer.update()
