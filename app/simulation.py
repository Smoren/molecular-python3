import random
import time
from typing import Tuple, Set, Dict, Callable

import numpy as np
import pygame
import numba as nb

from app.config import ATOMS_COLORS, MODE_DEBUG
from app.constants import A_COL_R, A_COL_Y, A_COL_X, A_COL_CX, A_COL_CY, A_COL_TYPE
from app.drawer import Drawer
from app.utils import interact_all, apply_speed


class Simulation:
    _screen: pygame.Surface
    _drawer: Drawer
    _clock: pygame.time.Clock
    _is_stopped: bool = False
    _atoms: np.ndarray
    _max_coord: np.ndarray
    _links: np.ndarray
    _atom_links: Dict[int, Set[Tuple[int, int]]]
    _draw_atoms_vectorized: Callable
    _draw_links_vectorized: Callable

    def __init__(self, atoms: np.ndarray, window_size: Tuple[int, int], max_coord: Tuple[int, int]):
        self._atoms = atoms
        self._max_coord = np.array(max_coord)
        self._screen = pygame.display.set_mode(window_size)
        self._drawer = Drawer(self._screen)
        self._clock = pygame.time.Clock()
        # TODO костыль: должен быть хотя бы один элемент
        self._links = np.array([[random.randint(0, 1000), random.randint(1000, 2000)] for _ in range(1)])
        self._atom_links = dict()
        self._draw_atoms_vectorized = np.vectorize(self._drawer.draw_circle)
        self._draw_links_vectorized = np.vectorize(self._draw_link)

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
        clusters_coords = np.unique(self._atoms[:, [A_COL_CX, A_COL_CY]], axis=0)
        interact_all(self._atoms, self._links, clusters_coords)

    def _step_move(self) -> None:
        apply_speed(self._atoms, self._max_coord)

    @nb.jit(
        forceobj=True,
        fastmath=True,
        looplift=True,
        boundscheck=False,
        parallel=True,
        cache=not MODE_DEBUG,
    )
    def _step_display(self) -> None:
        self._drawer.clear()

        colors = ATOMS_COLORS[self._atoms[:, A_COL_TYPE].astype(np.int64)]

        self._draw_atoms_vectorized(
            self._atoms[:, A_COL_X],
            self._atoms[:, A_COL_Y],
            self._atoms[:, A_COL_R],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )

        # for i in nb.prange(self._atoms.shape[0]):
        #     row = self._atoms[i]
        #     self._drawer.draw_circle(row[A_COL_X], row[A_COL_Y], row[A_COL_R], ATOMS_COLORS[int(row[A_COL_TYPE])])

        self._draw_links_vectorized(self._links[:, 0], self._links[:, 1])

        self._drawer.update()

    def _draw_link(self, lhs_idx, rhs_idx):
        lhs_color = ATOMS_COLORS[int(self._atoms[lhs_idx, A_COL_TYPE])]
        rhs_color = ATOMS_COLORS[int(self._atoms[rhs_idx, A_COL_TYPE])]
        self._drawer.draw_line(
            self._atoms[lhs_idx, [A_COL_X, A_COL_Y]],
            self._atoms[rhs_idx, [A_COL_X, A_COL_Y]],
            np.average([lhs_color, rhs_color], axis=0),
        )