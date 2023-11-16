from app.factories import generate_atoms
from app.simulation import Simulation

ATOMS_COUNT = 10000
WINDOW_SIZE = (1900, 1000)
CLUSTER_SIZE = 100

atoms = generate_atoms(ATOMS_COUNT, WINDOW_SIZE)

sim = Simulation(atoms, WINDOW_SIZE, WINDOW_SIZE, CLUSTER_SIZE)
sim.start()
