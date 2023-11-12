import random
from typing import Tuple

import pygame

from app.drawer import Drawer
from app.storage import Storage


class Ui:
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

            self._storage.move()
            self.display()
            self._clock.tick(30)

    def stop(self):
        self._is_stopped = True

    def display(self) -> None:
        self._drawer.clear()

        for _, row in self._storage.data.iterrows():
            self._drawer.draw_circle((row['x'], row['y']), row['radius'], (0, 0, 255))

        # self._drawer.draw_circle((250+random.randint(0, 10), 250), 75, (0, 0, 255))
        # self._drawer.draw_line((250, 250), (250, 400), (0, 0, 255))
        self._drawer.update()
