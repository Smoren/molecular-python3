from app.storage import Storage
from app.simulation import Simulation

ATOMS_COUNT = 10000
WINDOW_SIZE = (1900, 1000)
CLUSTER_SIZE = 100

storage = Storage(ATOMS_COUNT, WINDOW_SIZE, CLUSTER_SIZE)
ui = Simulation(WINDOW_SIZE, storage)
ui.start()
