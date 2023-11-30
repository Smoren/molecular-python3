import time
from typing import Tuple, Callable

import numpy as np
import numba as nb
import pygame

from app.config import ATOMS_COLORS, MAX_INTERACTION_DISTANCE, ATOMS_LJ_PARAMS, DELTA_T, GRAVITY, INERTIA
from app.constants import A_COL_R, A_COL_Y, A_COL_X, A_COL_CX, A_COL_CY, A_COL_TYPE
from app.screen import Screen
from app.logic import interact_atoms, apply_next_values, calc_next_positions


class Simulation:
    _screen: Screen
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
        self._screen = Screen(pygame.display.set_mode(window_size))
        self._clock = pygame.time.Clock()
        self._links = np.empty(shape=(0, 3), dtype=np.int64)

    def start(self):
        self._is_stopped = False
        i, time_sum, time_avg_size = 0, 0, 10
        while not self._is_stopped:
            ts = time.time_ns()

            self._handle_events()
            self._step()

            time_sum += time.time_ns() - ts
            i += 1
            if i == time_avg_size:
                step_time = round(time_sum / time_avg_size / 1_000_000)
                print(f'step spent: {step_time} | links: {self._links.shape[0]}')
                i, time_sum = 0, 0

    def stop(self):
        self._is_stopped = True

    def _step(self):
        self._step_calc_next_positions()
        self._step_interact_atoms()
        self._step_move()
        self._step_display()
        self._clock.tick(30)

    def _step_calc_next_positions(self) -> None:
        calc_next_positions(self._atoms, DELTA_T)

    def _step_interact_atoms(self) -> None:
        clusters_coords = np.unique(self._atoms[:, [A_COL_CX, A_COL_CY]], axis=0)
        interact_atoms(
            self._atoms, clusters_coords,
            MAX_INTERACTION_DISTANCE, ATOMS_LJ_PARAMS, GRAVITY, DELTA_T,
        )
        # interact_atoms.parallel_diagnostics(level=4)

    def _step_move(self) -> None:
        apply_next_values(self._atoms, MAX_INTERACTION_DISTANCE, INERTIA)

    def _step_display(self) -> None:
        self._screen.clear()
        self._display_atoms()
        self._screen.update()

    def _display_atoms(self):
        colors = ATOMS_COLORS[self._atoms[:, A_COL_TYPE].astype(np.int64)]
        self._screen.draw_circles_vectorized(
            self._atoms[:, A_COL_X],
            self._atoms[:, A_COL_Y],
            self._atoms[:, A_COL_R],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._screen.move_offset(100, 0)
                elif event.key == pygame.K_RIGHT:
                    self._screen.move_offset(-100, 0)
                elif event.key == pygame.K_UP:
                    self._screen.move_offset(0, 100)
                elif event.key == pygame.K_DOWN:
                    self._screen.move_offset(0, -100)
                elif event.key == pygame.K_EQUALS:
                    self._screen.move_scale(1.1)
                elif event.key == pygame.K_MINUS:
                    self._screen.move_scale(0.9)
                elif event.key == pygame.K_w:
                    ATOMS_LJ_PARAMS[0][0] += 0.1
                    print(f'sigma: {ATOMS_LJ_PARAMS[0][0]}')
                elif event.key == pygame.K_q:
                    ATOMS_LJ_PARAMS[0][0] -= 0.1
                    print(f'sigma: {ATOMS_LJ_PARAMS[0][0]}')
                elif event.key == pygame.K_s:
                    ATOMS_LJ_PARAMS[0][1] += 0.1
                    print(f'eps: {ATOMS_LJ_PARAMS[0][1]}')
                elif event.key == pygame.K_a:
                    ATOMS_LJ_PARAMS[0][1] -= 0.1
                    print(f'eps: {ATOMS_LJ_PARAMS[0][1]}')
