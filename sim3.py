import math
from collections import defaultdict
from typing import Tuple, List, Dict, Set

import numpy as np
import pygame


class Atom:
    x: int
    y: int
    vx: int
    vy: int
    radius: int
    cluster_coords: Tuple[int, int] | None = None

    def get_mass(self):
        return math.pi * self.radius * self.radius

    def __init__(self, x: int, y: int, vx: int, vy: int, radius: int):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius


def step(screen: pygame.Surface, atoms: List[Atom], clusters: Dict[Tuple[int, int], Set[Atom]]) -> None:
    for atom in atoms:
        cluster_coords = (math.floor(atom.x / 300), math.floor(atom.y / 300))

        if cluster_coords != atom.cluster_coords:
            if atom.cluster_coords is not None:
                clusters[atom.cluster_coords].remove(atom)
            clusters[cluster_coords].add(atom)
            atom.cluster_coords = cluster_coords

        atom.x += atom.vx
        atom.y += atom.vy

        if atom.x < 0 or atom.x > screen.get_width():
            atom.vx *= -1
        if atom.y < 0 or atom.y > screen.get_height():
            atom.vy *= -1

    for cluster_coords in clusters:
        cluster = clusters[cluster_coords]
        for lhs in cluster:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbour_cluster_coords = (
                        cluster_coords[0] + i,
                        cluster_coords[1] + j,
                    )

                    if neighbour_cluster_coords not in clusters:
                        continue

                    for rhs in clusters[neighbour_cluster_coords]:
                        if lhs == rhs:
                            continue

                        dx, dy = lhs.x - rhs.x, lhs.y - rhs.y
                        dist = math.sqrt(dx*dx + dy*dy)
                        radius_sum = lhs.radius + rhs.radius

                        if dist < radius_sum:
                            delta_x = -rhs.x + lhs.x
                            delta_y = -rhs.y + lhs.y
                            a = 0.01 * rhs.get_mass() / dist / dist

                            lhs.vx += delta_x * (radius_sum - dist) * a
                            lhs.vy += delta_y * (radius_sum - dist) * a
                        else:
                            delta_x = rhs.x - lhs.x
                            delta_y = rhs.y - lhs.y
                            a = 2 * rhs.get_mass() / dist / dist

                            lhs.vx += delta_x / dist * a
                            lhs.vy += delta_y / dist * a

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
    atoms_count = 1000

    atoms = generate_atoms(atoms_count, window_size)
    clusters: Dict[Tuple[int, int], Set[Atom]] = defaultdict(set)

    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    is_stopped = False
    while not is_stopped:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_stopped = True

        step(screen, atoms, clusters)
        clock.tick(30)
