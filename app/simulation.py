import time
from typing import Tuple, Callable

import pygame
import torch

from app.config import ATOMS_COLORS, MAX_INTERACTION_DISTANCE, ATOMS_GRAVITY, ATOMS_LINK_GRAVITY, ATOMS_LINKS, \
    ATOMS_LINK_TYPES, FORCE_BOUNCE_ELASTIC, FORCE_NOT_LINKED_GRAVITY, FORCE_LINKED_GRAVITY, MIN_LINK_DISTANCE, \
    FORCE_LINKED_ELASTIC, MAX_LINK_DISTANCE, INERTIAL_FACTOR, SIMULATION_SPEED
from app.constants import A_COL_R, A_COL_Y, A_COL_X, A_COL_CX, A_COL_CY, A_COL_TYPE, L_COL_LHS, L_COL_RHS
from app.screen import Screen
from app.logic import interact_atoms, apply_speed, interact_links


class Simulation:
    _screen: Screen
    _clock: pygame.time.Clock
    _is_stopped: bool = False
    _atoms: torch.Tensor
    _max_coord: torch.Tensor
    _links: torch.Tensor
    _display_atoms_vectorized: Callable
    _display_links_vectorized: Callable

    def __init__(self, atoms: torch.Tensor, window_size: Tuple[int, int], max_coord: Tuple[int, int]):
        self._atoms = atoms
        self._max_coord = torch.tensor(max_coord)
        self._screen = Screen(pygame.display.set_mode(window_size))
        self._clock = pygame.time.Clock()
        self._links = torch.empty(size=(0, 3), dtype=torch.int64)

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
        self._step_move()
        self._step_interact_atoms()
        self._step_interact_links()
        self._step_display()
        self._clock.tick(30)

    def _step_interact_atoms(self) -> None:
        clusters_coords = torch.unique(self._atoms[:, [A_COL_CX, A_COL_CY]], dim=0)
        new_links = interact_atoms(
            self._atoms, self._links, clusters_coords,
            MAX_INTERACTION_DISTANCE, ATOMS_GRAVITY, ATOMS_LINK_GRAVITY, ATOMS_LINKS, ATOMS_LINK_TYPES,
            FORCE_BOUNCE_ELASTIC, FORCE_NOT_LINKED_GRAVITY, FORCE_LINKED_GRAVITY,
            FORCE_LINKED_ELASTIC, MIN_LINK_DISTANCE,
        )
        # interact_atoms.parallel_diagnostics(level=4)
        self._links = torch.concat([self._links, new_links])

    def _step_interact_links(self) -> None:
        self._links = interact_links(self._atoms, self._links, MAX_LINK_DISTANCE)

    def _step_move(self) -> None:
        apply_speed(self._atoms, self._max_coord, MAX_INTERACTION_DISTANCE, INERTIAL_FACTOR, SIMULATION_SPEED)

    def _step_display(self) -> None:
        self._screen.clear()

        self._display_links()
        self._display_atoms()

        self._screen.update()

    def _display_atoms(self):
        colors = ATOMS_COLORS[self._atoms[:, A_COL_TYPE].to(torch.int64)]
        self._screen.draw_circles_vectorized(
            self._atoms[:, A_COL_X],
            self._atoms[:, A_COL_Y],
            self._atoms[:, A_COL_R],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )

    def _display_links(self):
        if self._links.shape[0] == 0:
            return

        lhs_coord_x = self._atoms[self._links[:, L_COL_LHS], A_COL_X]
        lhs_coord_y = self._atoms[self._links[:, L_COL_LHS], A_COL_Y]
        rhs_coord_x = self._atoms[self._links[:, L_COL_RHS], A_COL_X]
        rhs_coord_y = self._atoms[self._links[:, L_COL_RHS], A_COL_Y]

        lhs_colors = ATOMS_COLORS[self._atoms[self._links[:, L_COL_LHS], A_COL_TYPE].to(torch.int64)]
        rhs_colors = ATOMS_COLORS[self._atoms[self._links[:, L_COL_RHS], A_COL_TYPE].to(torch.int64)]
        colors = torch.mean(torch.stack((lhs_colors.to(torch.float64), rhs_colors.to(torch.float64))), dim=0)

        self._screen.draw_lines_vectorized(
            lhs_coord_x,
            lhs_coord_y,
            rhs_coord_x,
            rhs_coord_y,
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
