import time
from typing import Tuple

import pygame
import numba as nb

from app.constants import COL_R, COL_Y, COL_X
from app.drawer import Drawer
from app.storage import Storage


class Simulation:
    _screen: pygame.Surface
    _drawer: Drawer
    _clock: pygame.time.Clock
    _is_stopped: bool = False
    _storage: Storage

    def __init__(self, window_size: Tuple[int, int], storage: Storage):
        self._screen = pygame.display.set_mode(window_size)
        self._drawer = Drawer(self._screen)
        self._clock = pygame.time.Clock()
        self._storage = storage

    def start(self):
        self._is_stopped = False
        while not self._is_stopped:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()

            ts = time.time_ns()
            self._storage.move()
            self._storage.interact()
            self.display()
            self._clock.tick(30)
            print(f'step spent: {(time.time_ns() - ts) / 1_000_000}')

    def stop(self):
        self._is_stopped = True

    @nb.jit(
        forceobj=True,
        fastmath=True,
        looplift=True,
        boundscheck=False,
        parallel=True,
    )
    def display(self) -> None:
        self._drawer.clear()

        for i in nb.prange(self._storage.data.shape[0]):
            row = self._storage.data[i]
            self._drawer.draw_circle((row[COL_X], row[COL_Y]), row[COL_R], (0, 0, 255))
            i += 1
            if i > 10000:
                break

        self._drawer.update()
