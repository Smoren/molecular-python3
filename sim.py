from app.storage import Storage
from app.ui import Ui

ATOMS_COUNT = 5000
WINDOW_SIZE = (1900, 1000)
CLUSTER_SIZE = 100

storage = Storage(ATOMS_COUNT, WINDOW_SIZE, CLUSTER_SIZE)
ui = Ui(WINDOW_SIZE, storage)
ui.start()
