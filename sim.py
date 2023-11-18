from app.config import ATOMS_COUNT, WINDOW_SIZE
from app.factories import generate_atoms, generate_debug
from app.simulation import Simulation

atoms = generate_atoms(ATOMS_COUNT, WINDOW_SIZE)
# atoms = generate_debug()

sim = Simulation(atoms, WINDOW_SIZE, WINDOW_SIZE)
sim.start()
