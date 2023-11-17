from typing import Tuple

import numpy as np
import pygame


class Drawer:
    _screen: pygame.Surface
    _background_color: Tuple[int, int, int]

    def __init__(self, screen: pygame.Surface, background_color: Tuple[int, int, int] = (0, 0, 0)):
        self._screen = screen
        self._background_color = background_color

    def clear(self) -> None:
        self._screen.fill(self._background_color)

    def draw_circle(self, center_x: float, center_y: float, radius: int, color_r: int, color_g: int, color_b: int) -> None:
        pygame.draw.circle(self._screen, (color_r, color_g, color_b), (center_x, center_y), radius)

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        pygame.draw.line(self._screen, color, start, end)

    @staticmethod
    def update() -> None:
        pygame.display.flip()
