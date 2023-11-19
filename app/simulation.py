import time
from typing import Tuple, Callable

import numpy as np
import pygame
import numba as nb

from app.config import ATOMS_COLORS, MODE_DEBUG
from app.constants import A_COL_R, A_COL_Y, A_COL_X, A_COL_CX, A_COL_CY, A_COL_TYPE, L_COL_LHS, L_COL_RHS
from app.drawer import Drawer
from app.utils import interact_atoms, apply_speed, interact_links


class Simulation:
    _screen: pygame.Surface
    _drawer: Drawer
    _clock: pygame.time.Clock
    _is_stopped: bool = False
    _atoms: np.ndarray
    _max_coord: np.ndarray
    _links: np.ndarray
    _display_atoms_vectorized: Callable
    _display_links_vectorized: Callable

    def __init__(self, atoms: np.ndarray, window_size: Tuple[int, int], max_coord: Tuple[int, int]):
        self._atoms = atoms
        self._max_coord = np.array(max_coord)
        self._screen = pygame.display.set_mode(window_size)
        self._drawer = Drawer(self._screen)
        self._clock = pygame.time.Clock()
        self._links = np.empty(shape=(0, 3), dtype=np.int64)

    def start(self):
        self._is_stopped = False
        i, time_sum, time_avg_size = 0, 0, 10
        while not self._is_stopped:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()

            ts = time.time_ns()
            self._step_move()
            self._step_interact_atoms()
            self._step_interact_links()
            self._step_display()
            self._clock.tick(30)

            time_sum += time.time_ns() - ts
            i += 1
            if i == time_avg_size:
                step_time = round(time_sum / time_avg_size / 1_000_000)
                print(f'step spent: {step_time} | links: {self._links.shape[0]}')
                i, time_sum = 0, 0

    def stop(self):
        self._is_stopped = True

    def _step_interact_atoms(self) -> None:
        clusters_coords = np.unique(self._atoms[:, [A_COL_CX, A_COL_CY]], axis=0)
        new_links = interact_atoms(self._atoms, self._links, clusters_coords)
        # interact_atoms.parallel_diagnostics(level=4)
        self._links = np.append(self._links, new_links, axis=0)

    def _step_interact_links(self) -> None:
        self._links = interact_links(self._atoms, self._links)

    def _step_move(self) -> None:
        apply_speed(self._atoms, self._max_coord)

    @nb.jit(
        forceobj=True,
        fastmath=True,
        looplift=True,
        boundscheck=False,
        # parallel=True,
        cache=not MODE_DEBUG,
    )
    def _step_display(self) -> None:
        self._drawer.clear()

        self._display_links()
        self._display_atoms()

        self._drawer.update()

    def _display_atoms(self):
        colors = ATOMS_COLORS[self._atoms[:, A_COL_TYPE].astype(np.int64)]
        self._drawer.draw_circles_vectorized(
            self._atoms[:, A_COL_X],
            self._atoms[:, A_COL_Y],
            self._atoms[:, A_COL_R],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )

    # @nb.jit(parallel=True)
    def _display_links(self):
        if self._links.shape[0] == 0:
            return

        lhs_coords = self._atoms[self._links[:, L_COL_LHS]][:, [A_COL_X, A_COL_Y]]
        rhs_coords = self._atoms[self._links[:, L_COL_RHS]][:, [A_COL_X, A_COL_Y]]

        lhs_colors = ATOMS_COLORS[self._atoms[self._links[:, L_COL_LHS], A_COL_TYPE].astype(np.int64)]
        rhs_colors = ATOMS_COLORS[self._atoms[self._links[:, L_COL_RHS], A_COL_TYPE].astype(np.int64)]
        colors = np.average([lhs_colors, rhs_colors], axis=0)

        # self._display_links_vectorized(lhs_coords, rhs_coords, colors)

        for i in nb.prange(self._links.shape[0]):
            self._drawer.draw_line(lhs_coords[i], rhs_coords[i], colors[i])
