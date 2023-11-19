from app.config import ATOMS_COUNT, WINDOW_SIZE, MAX_COORD
from app.factories import generate_atoms
from app.simulation import Simulation

atoms = generate_atoms(ATOMS_COUNT, MAX_COORD)
sim = Simulation(atoms, WINDOW_SIZE, MAX_COORD)
sim.start()
