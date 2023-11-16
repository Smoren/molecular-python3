from app.factories import generate_atoms
from app.simulation import Simulation

ATOMS_COUNT = 3000
WINDOW_SIZE = (1900, 1000)

atoms = generate_atoms(ATOMS_COUNT, WINDOW_SIZE)

sim = Simulation(atoms, WINDOW_SIZE, WINDOW_SIZE)
sim.start()
