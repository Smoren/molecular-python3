from typing import Tuple, Callable, List

import numpy as np
import pygame


class Screen:
    draw_circles_vectorized: Callable
    draw_lines_vectorized: Callable
    _screen: pygame.Surface
    _background_color: Tuple[int, int, int]
    _offset: List[int]
    _scale: float

    def __init__(self, screen: pygame.Surface, background_color: Tuple[int, int, int] = (0, 0, 0)):
        self._screen = screen
        self._offset = [0, 0]
        self._scale = 1
        self._background_color = background_color
        self.draw_circles_vectorized = np.vectorize(self.draw_circle)
        self.draw_lines_vectorized = np.vectorize(self.draw_line)

    def move_offset(self, dx: int, dy: int):
        self._offset[0] += dx
        self._offset[1] += dy

    def move_scale(self, dv):
        self._scale *= dv

    def clear(self) -> None:
        self._screen.fill(self._background_color)

    def draw_circle(self, center_x: float, center_y: float, radius: int, color_r: int, color_g: int, color_b: int) -> None:
        pygame.draw.circle(
            self._screen,
            (color_r, color_g, color_b),
            ((center_x + self._offset[0])*self._scale, (center_y + self._offset[1])*self._scale),
            radius*self._scale
        )

    def draw_line(self, start_x: float, start_y: float, end_x: float, end_y: float, color_r: int, color_g: int, color_b: int) -> None:
        pygame.draw.line(
            self._screen,
            (color_r, color_g, color_b),
            ((start_x + self._offset[0])*self._scale, (start_y + self._offset[1])*self._scale),
            ((end_x + self._offset[0])*self._scale, (end_y + self._offset[1])*self._scale),
        )

    @staticmethod
    def update() -> None:
        pygame.display.flip()
