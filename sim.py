from app.config import ATOMS_COUNT, WINDOW_SIZE
from app.factories import generate_atoms
from app.simulation import Simulation

atoms = generate_atoms(ATOMS_COUNT, WINDOW_SIZE)

sim = Simulation(atoms, WINDOW_SIZE, WINDOW_SIZE)
sim.start()
