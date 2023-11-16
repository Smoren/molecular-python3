import math
from typing import Tuple, List

import numpy as np
import pygame

from utils import calc_speed_delta


class Atom:
    x: int
    y: int
    vx: int
    vy: int
    radius: int

    def get_mass(self):
        return math.pi * self.radius * self.radius

    def __init__(self, x: int, y: int, vx: int, vy: int, radius: int):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius


def step(screen: pygame.Surface, atoms: List[Atom]) -> None:
    screen_width, screen_height = screen.get_width(), screen.get_height()

    for atom in atoms:
        atom.x += atom.vx
        atom.y += atom.vy

        if atom.x < 0 or atom.x > screen_width:
            atom.vx *= -1
        if atom.y < 0 or atom.y > screen_height:
            atom.vy *= -1

    for lhs in atoms:
        for rhs in atoms:
            if lhs == rhs:
                continue

            delta_vx, delta_vy = calc_speed_delta(lhs.x, rhs.x, lhs.y, rhs.y, lhs.radius, rhs.radius)
            lhs.vx += delta_vx
            lhs.vy += delta_vy

    screen.fill((0, 0, 0))
    for atom in atoms:
        pygame.draw.circle(screen, (255, 255, 255), (atom.x, atom.y), atom.radius)
    pygame.display.flip()


def generate_atoms(atoms_count: int, window_size: Tuple[int, int]) -> List[Atom]:
    atoms = []

    atoms.append(Atom(500, 500, 0, 0, 50))
    atoms.append(Atom(700, 500, 0, 10, 5))

    for _ in range(atoms_count):
        x = np.random.randint(low=0, high=window_size[0])
        y = np.random.randint(low=0, high=window_size[1])
        vx = np.random.randint(low=-10, high=10)
        vy = np.random.randint(low=-10, high=10)
        radius = np.random.randint(low=5, high=10)
        atoms.append(Atom(x, y, vx, vy, radius))

    return atoms


if __name__ == '__main__':
    window_size = (1900, 1000)
    atoms_count = 500
    atoms = generate_atoms(atoms_count, window_size)
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    is_stopped = False
    while not is_stopped:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_stopped = True

        step(screen, atoms)
        clock.tick(30)
