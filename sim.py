import numpy as np

from app.config import ATOMS_COUNT, WINDOW_SIZE, MAX_COORD
from app.factories import generate_random, generate_vessel
from app.simulation import Simulation

atoms = generate_random(ATOMS_COUNT, MAX_COORD, 0)
# vessel = generate_vessel(1000, MAX_COORD)
# all = np.concatenate((atoms, vessel), axis=0)

sim = Simulation(atoms, WINDOW_SIZE, MAX_COORD)
sim.start()
