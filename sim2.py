from typing import Tuple, List

import numpy as np
import pygame


class Atom:
    x: int
    y: int
    vx: int
    vy: int
    radius: int

    def __init__(self, x: int, y: int, vx: int, vy: int, radius: int):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius


def step(screen: pygame.Surface, atoms: List[Atom]) -> None:
    for atom in atoms:
        atom.x += atom.vx
        atom.y += atom.vy

        if atom.x < 0 or atom.x > screen.get_width():
            atom.vx *= -1
        if atom.y < 0 or atom.y > screen.get_height():
            atom.vy *= -1

    for lhs in atoms:
        for rhs in atoms:
            if lhs == rhs:
                continue
            dist = np.linalg.norm(np.array([lhs.x - rhs.x, lhs.y - rhs.y]))
            if dist < lhs.radius + rhs.radius:
                delta_x = lhs.x - rhs.x
                delta_y = lhs.y - rhs.y
                lhs.vx += delta_x / dist * 2
                lhs.vy += delta_y / dist * 2
            else:
                G = 30
                force = G * lhs.radius * rhs.radius / dist**2
                delta_x = lhs.x - rhs.x
                delta_y = lhs.y - rhs.y
                force_x = force * delta_x / dist
                force_y = force * delta_y / dist
                lhs.x -= force_x
                lhs.y -= force_y

    screen.fill((0, 0, 0))
    for atom in atoms:
        pygame.draw.circle(screen, (255, 255, 255), (atom.x, atom.y), atom.radius)
    pygame.display.flip()


def generate_atoms(atoms_count: int, window_size: Tuple[int, int]) -> List[Atom]:
    atoms = []
    for _ in range(atoms_count):
        x = np.random.randint(low=0, high=window_size[0])
        y = np.random.randint(low=0, high=window_size[1])
        vx = np.random.randint(low=-10, high=10)
        vy = np.random.randint(low=-10, high=10)
        radius = np.random.randint(low=5, high=25)
        atoms.append(Atom(x, y, vx, vy, radius))

    return atoms


if __name__ == '__main__':
    window_size = (1900, 1000)
    atoms_count = 10
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
